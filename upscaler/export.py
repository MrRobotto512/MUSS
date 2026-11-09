"""Segmented, resumable export.

The frame range is split into segments of a few seconds. Each GPU worker repeatedly takes the
next unfinished segment and runs it through its own pipeline:

    NVDEC decode (on that GPU) -> AI model -> RGB->YUV on the GPU -> NVENC (on that GPU) -> seg_XXXXX.mkv

A segment file only gets its final name once it is completely encoded, so after a crash, power
cut or Stop, running the same export again skips every finished segment ("resume"). job.json
records the exact settings; if anything that affects the picture changed, the old parts are
discarded instead of being mixed in. When all segments exist they are joined without
re-encoding and the original audio is added.
"""
import collections
import json
import os
import shutil
import subprocess
import threading
import time

import torch

from . import gpu_io
from .engine import (ENCODERS, MP4_SAFE_AUDIO, NO_WINDOW, Progress, _in_matrix, find_tool, get_nets, probe,
                     upscale_tensor)
from .models import ModelSpec

VERSION = 1
SEGMENT_SECONDS = 4.0


def parts_dir(output_path: str) -> str:
    return output_path + ".parts"


def resumable_progress(output_path: str):
    """(done_segments, total_segments) if an unfinished export exists for this output, else None."""
    d = parts_dir(output_path)
    try:
        with open(os.path.join(d, "job.json"), encoding="utf-8") as f:
            m = json.load(f)
        done = sum(1 for i in range(m["segments"]) if os.path.exists(os.path.join(d, f"seg_{i:05d}.mkv")))
        return done, m["segments"]
    except (OSError, ValueError, KeyError):
        return None


class ExportJob:
    def __init__(self, input_path, output_path, spec: ModelSpec, gpu_ids, encoder_label, cq=18,
                 start_frame=0, frame_count=None, use_tensorrt=False, on_log=print, on_progress=None,
                 remote_workers=None):
        self.input_path, self.output_path, self.spec = input_path, output_path, spec
        self.gpu_ids = list(gpu_ids)
        self.encoder_label, self.cq = encoder_label, cq
        self.start_frame, self.frame_count = max(int(start_frame), 0), frame_count
        self.use_tensorrt = use_tensorrt
        self.log = on_log
        self.on_progress = on_progress or (lambda p: None)
        self.remote_workers = list(remote_workers or [])
        self.cancel_event = threading.Event()
        self._error = None
        self._lock = threading.Lock()
        self._procs = set()
        self.resumed_segments = 0

    # ------------------------------------------------------------------ public
    def cancel(self):
        first = not self.cancel_event.is_set()
        self.cancel_event.set()
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            if p.poll() is None:
                p.kill()
        if first:
            for rw in self.remote_workers:          # tell render nodes to stop their current segment
                if hasattr(rw, "cancel"):
                    threading.Thread(target=rw.cancel, args=(self,), daemon=True).start()

    def run(self) -> bool:
        """Blocking. True = finished, False = cancelled (finished segments are kept for resume)."""
        if not self.gpu_ids and not self.remote_workers:
            raise RuntimeError("Select at least one GPU.")
        info = self.info = probe(self.input_path)
        fps = float(info.fps)
        w, h = info.width, info.height
        total = max(info.frames - self.start_frame, 0)
        if self.frame_count:
            total = min(total, int(self.frame_count))
        if total <= 0:
            raise RuntimeError("Nothing to export in this range.")
        seg_frames = max(int(round(fps * SEGMENT_SECONDS)), 24)
        segs = [(i, self.start_frame + i * seg_frames, min(seg_frames, total - i * seg_frames))
                for i in range((total + seg_frames - 1) // seg_frames)]
        self.total_frames, self.segments = total, segs

        self.workdir = parts_dir(self.output_path)
        self._prepare_workdir(info, total, seg_frames, len(segs))
        done = [s for s in segs if os.path.exists(self._seg_path(s[0]))]
        pending = collections.deque(s for s in segs if not os.path.exists(self._seg_path(s[0])))
        self.resumed_segments = len(done)
        done_frames = sum(s[2] for s in done)
        self.log(f"Input {w}x{h} @ {fps:.3f} fps, {total} frames in {len(segs)} segments -> {w * 2}x{h * 2}")
        if done:
            self.log(f"Resuming: {len(done)} of {len(segs)} segments ({done_frames} frames) were already rendered.")

        state = {"done": done_frames, "session": 0, "t0": time.time(), "last": 0.0}
        per_worker = collections.Counter()

        def report(force=False):
            now = time.time()
            if not force and now - state["last"] < 0.5:
                return
            state["last"] = now
            fps_now = state["session"] / max(now - state["t0"], 1e-6)
            remaining = total - state["done"]
            self.on_progress(Progress(state["done"], total, fps_now, remaining / fps_now if fps_now else 0,
                                      dict(per_worker)))

        def frame_done(worker_id):
            with self._lock:
                state["done"] += 1
                state["session"] += 1
                per_worker[worker_id] += 1
            report()

        def take():
            with self._lock:
                return pending.popleft() if pending and not self.cancel_event.is_set() else None

        def give_back(seg):
            with self._lock:
                pending.appendleft(seg)

        report(force=True)
        if pending:
            workers = []
            if self.gpu_ids:
                nets = get_nets(self.spec, self.gpu_ids, self.use_tensorrt, w, h, log=self.log,
                                cancel_event=self.cancel_event)
                if nets is None:
                    return False
                for n in nets.values():
                    if hasattr(n, "warm_up"):
                        n.warm_up(self.spec.workers_per_gpu)
                for gid in self.gpu_ids:
                    workers.append(threading.Thread(target=self._gpu_worker, daemon=True,
                                                    args=(gid, nets[gid], info, take, give_back, frame_done)))
            for rw in self.remote_workers:
                workers.append(threading.Thread(target=self._remote_worker, daemon=True,
                                                args=(rw, info, take, give_back, frame_done)))
            for t in workers:
                t.start()
            for t in workers:
                t.join()

        if self._error:
            raise RuntimeError(str(self._error))
        if self.cancel_event.is_set():
            n_done = sum(1 for s in segs if os.path.exists(self._seg_path(s[0])))
            self.log(f"Stopped. {n_done} of {len(segs)} segments are saved; export again to resume.")
            return False
        missing = [s[0] for s in segs if not os.path.exists(self._seg_path(s[0]))]
        if missing:
            raise RuntimeError(f"{len(missing)} segment(s) were not rendered.")
        report(force=True)
        self._join(info, total)
        shutil.rmtree(self.workdir, ignore_errors=True)
        self.log(f"Done: {total} frames. " + ", ".join(f"{k}: {v}" for k, v in per_worker.items()))
        return True

    # ------------------------------------------------------------------ workdir / resume
    def _manifest(self, info, total, seg_frames, n_segs):
        st = os.stat(self.input_path)
        model_size = os.path.getsize(self.spec.path) if self.spec.path and os.path.exists(self.spec.path) else 0
        return {
            "version": VERSION, "input": os.path.abspath(self.input_path), "input_size": st.st_size,
            "input_mtime": int(st.st_mtime), "model": self.spec.key, "model_size": model_size,
            "encoder": self.encoder_label, "cq": self.cq, "start_frame": self.start_frame, "frames": total,
            "segment_frames": seg_frames, "segments": n_segs, "fps": str(info.fps),
            "size": [info.width, info.height],
        }

    def _prepare_workdir(self, info, total, seg_frames, n_segs):
        want = self._manifest(info, total, seg_frames, n_segs)
        path = os.path.join(self.workdir, "job.json")
        try:
            with open(path, encoding="utf-8") as f:
                have = json.load(f)
        except (OSError, ValueError):
            have = None
        if have != want:
            if have is not None:
                self.log("Settings changed since the unfinished export; starting it again from the beginning.")
            shutil.rmtree(self.workdir, ignore_errors=True)
        os.makedirs(self.workdir, exist_ok=True)
        for f in os.listdir(self.workdir):          # half-written segments from an interruption
            if f.endswith(".tmp"):
                try:
                    os.remove(os.path.join(self.workdir, f))
                except OSError:
                    pass
        with open(path, "w", encoding="utf-8") as f:
            json.dump(want, f, indent=1)

    def _seg_path(self, i):
        return os.path.join(self.workdir, f"seg_{i:05d}.mkv")

    # ------------------------------------------------------------------ local GPU worker
    def _gpu_worker(self, gid, net, info, take, give_back, frame_done):
        reader = None
        try:
            reader = gpu_io.open_reader(self.input_path, info, gid, find_tool("ffmpeg"), _in_matrix(info),
                                        log=self.log)
            self.log(f"GPU {gid}: decoding on {reader.kind}")
            while True:
                seg = take()
                if seg is None:
                    return
                try:
                    self._render_segment(gid, net, reader, info, seg, lambda: frame_done(f"GPU {gid}"))
                except Exception:
                    if not self.cancel_event.is_set():
                        give_back(seg)
                    raise
        except Exception as e:  # noqa: BLE001
            self._fail(e, f"GPU {gid}")
        finally:
            if reader:
                reader.close()

    def _render_segment(self, gid, net, reader, info, seg, frame_done):
        i, start, count = seg
        w, h = info.width, info.height
        dev = torch.device("cuda", gid)
        codec, extra, pix = ENCODERS[self.encoder_label]
        ten_bit = pix == "p010le"
        enc = self._start_segment_encoder(i, gid, info, codec, extra, pix)
        n_inf = max(self.spec.workers_per_gpu, 1)
        import queue as _q
        in_q: _q.Queue = _q.Queue(maxsize=n_inf * 2)
        results, cond = {}, threading.Condition()
        nxt = {"i": 0}
        errors = []

        def put(item):
            while True:
                try:
                    in_q.put(item, timeout=0.5)
                    return True
                except _q.Full:
                    if self.cancel_event.is_set() or errors:
                        return False

        def decode():
            try:
                for k, src in enumerate(reader.frames(start, count)):
                    if self.cancel_event.is_set() or errors or not put((k, src)):
                        break
            except Exception as e:  # noqa: BLE001
                errors.append(e)
            finally:
                for _ in range(n_inf):
                    if not put(None):
                        break

        def infer():
            stream = torch.cuda.Stream(dev)
            try:
                while True:
                    item = in_q.get()
                    if item is None or self.cancel_event.is_set() or errors:
                        return
                    k, src = item
                    with torch.cuda.device(dev), torch.cuda.stream(stream), torch.inference_mode():
                        try:
                            y = upscale_tensor(net, self.spec, src, w, h, dev)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if getattr(net, "is_tensorrt", False):
                                raise RuntimeError(f"GPU {gid} ran out of memory with TensorRT. "
                                                   "Turn TensorRT off for this video.")
                            y = upscale_tensor(net, self.spec, src, w, h, dev, tile=512)
                        data = gpu_io.rgb_to_yuv420(y, ten_bit)
                    with cond:
                        while k - nxt["i"] > n_inf * 3 and not self.cancel_event.is_set() and not errors:
                            cond.wait(0.5)
                        results[k] = data
                        cond.notify_all()
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                with cond:
                    cond.notify_all()

        threads = [threading.Thread(target=decode, daemon=True)] + \
                  [threading.Thread(target=infer, daemon=True) for _ in range(n_inf)]
        for t in threads:
            t.start()
        written = 0
        try:
            while written < count:
                with cond:
                    while written not in results:
                        if self.cancel_event.is_set() or errors:
                            raise _Stop()
                        if not any(t.is_alive() for t in threads) and written not in results:
                            raise RuntimeError(f"Segment {i}: decoder delivered only {written} of {count} frames.")
                        cond.wait(0.25)
                    data = results.pop(written)
                    nxt["i"] = written + 1
                    cond.notify_all()
                enc.stdin.write(data)
                written += 1
                frame_done()
            enc.stdin.close()
            rc = enc.wait()
            if rc != 0:
                raise RuntimeError(f"Segment {i}: encoder failed (exit {rc}): {' '.join(enc.stderr_lines[-3:])}")
            os.replace(self._seg_path(i) + ".tmp", self._seg_path(i))
        except _Stop:
            if errors and not self.cancel_event.is_set():
                raise errors[0]
        finally:
            if enc.poll() is None:
                enc.kill()
            enc.wait()
            with self._lock:
                self._procs.discard(enc)
            for t in threads:
                t.join(timeout=2)

    def _start_segment_encoder(self, i, gid, info, codec, extra, pix):
        enc_gpu = gid
        if codec == "av1_nvenc" and torch.cuda.get_device_capability(gid) < (8, 9):
            # RTX 30 cards have no AV1 encoder: use the first card that has one (RTX 40+)
            capable = [g for g in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(g) >= (8, 9)]
            if not capable:
                raise RuntimeError("None of your GPUs can encode AV1 (needs an RTX 40 series card).")
            enc_gpu = capable[0]
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{info.width * 2}x{info.height * 2}",
               "-framerate", str(info.fps), "-i", "-",
               "-c:v", codec, "-gpu", str(enc_gpu), *extra, "-preset", "p6", "-tune", "hq", "-rc", "vbr",
               "-cq", str(self.cq), "-b:v", "0",
               "-color_range", "tv", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
               "-an", "-f", "matroska", self._seg_path(i) + ".tmp"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW, bufsize=0)
        p.stderr_lines = []

        def drain():
            for line in iter(p.stderr.readline, b""):
                p.stderr_lines.append(line.decode(errors="replace").rstrip())
        threading.Thread(target=drain, daemon=True).start()
        with self._lock:
            self._procs.add(p)
        return p

    # ------------------------------------------------------------------ remote worker (render node)
    def _remote_worker(self, node, info, take, give_back, frame_done):
        try:
            node.prepare(self, info)
            while True:
                seg = take()
                if seg is None:
                    return
                try:
                    node.render_segment(self, seg, self._seg_path(seg[0]), lambda: frame_done(node.name))
                except Exception as e:  # noqa: BLE001
                    give_back(seg)
                    if self.cancel_event.is_set():
                        return
                    # a lost render node shouldn't kill the export: its segment goes back to the queue
                    self.log(f"{node.name}: {e} - its work goes back to the other workers.")
                    return
        except Exception as e:  # noqa: BLE001
            self.log(f"{node.name}: not used ({e}).")

    # ------------------------------------------------------------------ join
    def _join(self, info, total):
        lst = os.path.join(self.workdir, "segments.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for i, _, _ in self.segments:
                f.write(f"file '{os.path.basename(self._seg_path(i))}'\n")
        ext = os.path.splitext(self.output_path)[1].lower()
        fmt = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska"}.get(ext, "mp4")
        is_mp4 = fmt in ("mp4", "mov")
        fps = float(info.fps)
        cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", lst]
        whole = self.start_frame == 0 and total >= info.frames
        if not whole:
            cmd += ["-ss", f"{self.start_frame / fps:.6f}", "-t", f"{total / fps:.6f}"]
        cmd += ["-i", self.input_path, "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c:v", "copy"]
        if info.audio_codec and (not is_mp4 or info.audio_codec in MP4_SAFE_AUDIO):
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "320k"]
        codec = ENCODERS[self.encoder_label][0]
        if is_mp4:
            if codec == "hevc_nvenc":
                cmd += ["-tag:v", "hvc1"]          # plays in QuickTime / Apple devices
            cmd += ["-movflags", "+faststart"]
        tmp = self.output_path + ".joining"
        cmd += ["-f", fmt, tmp]
        self.log(f"Joining {len(self.segments)} segments and adding audio...")
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError(f"Joining segments failed: {r.stderr.strip()[-400:]}")
        os.replace(tmp, self.output_path)

    def _fail(self, e, who):
        with self._lock:
            if self._error is None and not self.cancel_event.is_set():
                self._error = RuntimeError(f"{who}: {e}")
                self.log(f"Error ({who}): {e}")
        self.cancel()


class _Stop(Exception):
    pass
