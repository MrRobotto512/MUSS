"""Restore the high frequencies of a file's audio track (any length), on all GPUs, resumable.

Work folder (next to the output, `<output>.audio_parts`):
    src.f32      decoded audio, 48 kHz float32 interleaved (streamed to disk, not RAM)
    body.f32     the restored band for each chunk's non-overlapping middle part
    heads.f32 / tails.f32   each chunk's overlap regions, cross-faded at the end
    done.txt     finished chunk ids (writes are plain assignments, so redoing a chunk is harmless)
    job.json     settings; anything that changes the result starts the work over
"""
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import audio
from .engine import NO_WINDOW, find_tool

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".ac3", ".eac3", ".dts",
              ".mka", ".aif", ".aiff", ".alac", ".ape", ".wv", ".mp2")
VERSION = 1


@dataclass
class AudioInfo:
    codec: str
    channels: int
    layout: str
    sample_rate: int
    duration: float
    bitrate: int            # bits/s, 0 if unknown


def probe_audio(path: str) -> AudioInfo | None:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_streams",
                          "-show_format", "-of", "json", path], capture_output=True, text=True,
                         creationflags=NO_WINDOW).stdout
    try:
        d = json.loads(out)
    except ValueError:
        return None
    if not d.get("streams"):
        return None
    s, f = d["streams"][0], d.get("format", {})
    return AudioInfo(codec=s.get("codec_name", "?"), channels=int(s.get("channels") or 2),
                     layout=s.get("channel_layout") or "", sample_rate=int(s.get("sample_rate") or 48000),
                     duration=float(s.get("duration") or f.get("duration") or 0),
                     bitrate=int(s.get("bit_rate") or f.get("bit_rate") or 0))


def describe(ai: AudioInfo) -> str:
    rate = f"{ai.sample_rate / 1000:g} kHz"
    ch = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}.get(ai.channels, f"{ai.channels} ch")
    br = f" {ai.bitrate // 1000} kbps" if ai.bitrate else ""
    return f"{ai.codec.upper()}{br} {ch} {rate}"


def analyze_cutoff(path: str, info: AudioInfo, start=0.0, duration=None) -> float:
    """Where the treble stops, measured on up to 12 x 15 s excerpts spread over the range (mono mix)."""
    dur = duration or max(info.duration - start, 0)
    n_seg = 12 if dur > 240 else 1
    seg_len = 15.0 if n_seg > 1 else dur
    pieces = []
    for i in range(n_seg):
        t = start + (dur - seg_len) * (i + 0.5) / n_seg if n_seg > 1 else start
        raw = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}", "-t", f"{seg_len:.3f}",
                              "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(audio.SR), "-f", "f32le", "-"],
                             capture_output=True, creationflags=NO_WINDOW).stdout
        pieces.append(np.frombuffer(raw, np.float32))
    x = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    # the cutoff can't be above the file's own Nyquist frequency
    return min(audio.measure_cutoff(x), info.sample_rate / 2)


def lfe_index(info: AudioInfo):
    """Index of the subwoofer channel in ffmpeg's channel order, or None."""
    if "LFE" in info.layout.upper() or info.layout in ("5.1", "5.1(side)", "7.1", "7.1(wide)", "6.1"):
        return 3
    return 3 if info.channels in (6, 8) else None


class AudioRestoreJob:
    """Decode -> restore the band above the cutoff (all GPUs) -> write a float32 WAV of the result.

    run() returns the restored WAV path, or None if the audio is already full range (caller keeps the
    original audio). False = cancelled (finished chunks are kept for resume)."""

    def __init__(self, input_path, workdir, gpu_ids, steps=audio.DEFAULT_STEPS, start=0.0, duration=None,
                 on_log=print, on_progress=None, cancel_event=None):
        self.input_path, self.workdir = input_path, workdir
        self.gpu_ids = list(gpu_ids)
        self.steps, self.start, self.duration = steps, float(start or 0), duration
        self.log = on_log
        self.on_progress = on_progress or (lambda done, total, eta: None)
        self.cancel_event = cancel_event or threading.Event()
        self.cutoff = None
        self.resumed = 0

    def cancel(self):
        self.cancel_event.set()

    # ------------------------------------------------------------------ steps
    def _decode(self, info):
        path = os.path.join(self.workdir, "src.f32")
        if os.path.exists(path + ".ok"):
            return path
        cmd = [find_tool("ffmpeg"), "-v", "error", "-y"]
        if self.start:
            cmd += ["-ss", f"{self.start:.6f}"]
        cmd += ["-i", self.input_path]
        if self.duration:
            cmd += ["-t", f"{self.duration:.6f}"]
        cmd += ["-map", "0:a:0", "-ac", str(info.channels), "-ar", str(audio.SR), "-f", "f32le", path]
        self.log("Decoding the audio track...")
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError("Decoding the audio failed: " + r.stderr.strip()[-300:])
        open(path + ".ok", "w").close()
        return path

    def _prepare(self, info, n, cutoff, plan):
        want = {"version": VERSION, "input": os.path.abspath(self.input_path),
                "size": os.path.getsize(self.input_path), "mtime": int(os.path.getmtime(self.input_path)),
                "start": self.start, "duration": self.duration, "steps": self.steps, "cutoff": round(cutoff),
                "channels": info.channels, "samples": n, "chunks": len(plan)}
        jpath = os.path.join(self.workdir, "job.json")
        try:
            with open(jpath, encoding="utf-8") as f:
                have = json.load(f)
        except (OSError, ValueError):
            have = None
        if have != want:
            for f in ("body.f32", "heads.f32", "tails.f32", "done.txt", "restored.wav"):
                try:
                    os.remove(os.path.join(self.workdir, f))
                except OSError:
                    pass
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(want, f, indent=1)

    # ------------------------------------------------------------------ main
    def run(self):
        info = probe_audio(self.input_path)
        if info is None:
            raise RuntimeError("This file has no audio track.")
        os.makedirs(self.workdir, exist_ok=True)
        dur = self.duration or max(info.duration - self.start, 0)
        self.cutoff = analyze_cutoff(self.input_path, info, self.start, dur)
        if self.cutoff >= audio.FULL_RANGE_HZ:
            self.log(f"Audio is already full range (treble reaches {self.cutoff / 1000:.1f} kHz): nothing to restore.")
            return None
        self.log(f"Audio treble stops at {self.cutoff / 1000:.1f} kHz ({describe(info)}): restoring up to 24 kHz.")
        if not audio.weights_present():
            raise RuntimeError("The AudioSR model hasn't been downloaded yet.")

        src_path = self._decode(info)
        ch = info.channels
        n = os.path.getsize(src_path) // (4 * ch)
        src = np.memmap(src_path, np.float32, "r", shape=(n, ch))
        plan = audio.chunk_plan(n)
        lfe = lfe_index(info)
        chans = [c for c in range(ch) if c != lfe]
        self._prepare(info, n, self.cutoff, plan)
        ov = int(audio.OVERLAP * audio.SR)
        n_items = len(chans) * len(plan)

        def mm(name, shape):
            p = os.path.join(self.workdir, name)
            return np.memmap(p, np.float32, "r+" if os.path.exists(p) else "w+", shape=shape)

        body = mm("body.f32", (n, ch))
        heads = mm("heads.f32", (ch, len(plan), ov))
        tails = mm("tails.f32", (ch, len(plan), ov))
        done_path = os.path.join(self.workdir, "done.txt")
        done = set()
        if os.path.exists(done_path):
            done = {int(l) for l in open(done_path) if l.strip().isdigit()}
        self.resumed = len(done)
        if done:
            self.log(f"Resuming audio: {len(done)} of {n_items} chunks already restored.")
        todo = [(c, k) for c in chans for k in range(len(plan)) if c * len(plan) + k not in done]
        lock = threading.Lock()
        errors = []
        t0 = time.time()
        stats = {"done": len(done), "session": 0}

        def worker(gid):
            try:
                model = audio.load_model(gid, log=self.log)
                while not self.cancel_event.is_set():
                    with lock:
                        if not todo:
                            return
                        c, k = todo.pop(0)
                    s, length = plan[k]
                    seg = np.array(src[s:s + length, c], dtype=np.float32)
                    hb = audio.restore_chunk(model, seg, self.cutoff, self.steps, seed=42 + k)
                    first, last = k == 0, k == len(plan) - 1
                    a = 0 if first else ov                          # body = everything but the overlaps
                    b = length if last else max(length - ov, a)
                    with lock:
                        body[s + a:s + b, c] = hb[a:b]
                        if not first:
                            heads[c, k, :min(ov, length)] = hb[:min(ov, length)]
                        if not last:
                            tails[c, k, :] = hb[length - ov:length]
                        with open(done_path, "a") as f:
                            f.write(f"{c * len(plan) + k}\n")
                        stats["done"] += 1
                        stats["session"] += 1
                        rate = stats["session"] / max(time.time() - t0, 1e-6)
                        self.on_progress(stats["done"], n_items, (n_items - stats["done"]) / rate if rate else 0)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                self.cancel_event.set()

        threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in self.gpu_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        body.flush()
        heads.flush()
        tails.flush()
        if errors:
            raise RuntimeError(f"Audio restoration failed: {errors[0]}")
        if self.cancel_event.is_set():
            return False
        return self._assemble(src, body, heads, tails, plan, chans, n, ch, ov)

    def _assemble(self, src, body, heads, tails, plan, chans, n, ch, ov):
        """original + restored band, overlaps cross-faded, peak-guarded; written as float32 WAV."""
        self.log("Assembling the restored audio...")
        fade = np.hanning(2 * ov).astype(np.float32)
        fin, fout = fade[:ov], fade[ov:]
        out_path = os.path.join(self.workdir, "restored.f32")
        out = np.memmap(out_path, np.float32, "w+", shape=(n, ch))
        block = audio.SR * 60
        peak_src, peak_out = 0.0, 0.0
        for s in range(0, n, block):                    # original + bodies, a minute at a time
            e = min(s + block, n)
            a = np.asarray(src[s:e])
            out[s:e] = a + body[s:e]
            peak_src = max(peak_src, float(np.abs(a).max()))
        for c in chans:                                 # cross-faded overlaps
            for k in range(1, len(plan)):
                s = plan[k][0]
                m = min(ov, n - s)
                out[s:s + m, c] += tails[c, k - 1, :m] * fout[:m] + heads[c, k, :m] * fin[:m]
        for s in range(0, n, block):
            peak_out = max(peak_out, float(np.abs(out[s:min(s + block, n)]).max()))
        limit = 0.999                                   # integer formats (FLAC/ALAC/AAC/PCM) clip at full scale
        if peak_out > limit:                            # added treble must not cause clipping
            g = limit / peak_out
            self.log(f"Lowering the level by {20 * np.log10(g):.1f} dB to avoid clipping.")
            for s in range(0, n, block):
                out[s:min(s + block, n)] *= g
        out.flush()
        wav = os.path.join(self.workdir, "restored.wav")
        r = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-y", "-f", "f32le", "-ar", str(audio.SR),
                            "-ac", str(ch), "-i", out_path, "-c:a", "pcm_f32le", wav],
                           capture_output=True, text=True, creationflags=NO_WINDOW)
        del out
        os.remove(out_path)
        if r.returncode != 0:
            raise RuntimeError("Writing the restored audio failed: " + r.stderr.strip()[-300:])
        return wav


def audio_codec_args(container_ext: str, channels: int) -> list:
    """Best encoder for the restored audio in each container."""
    ext = container_ext.lower().lstrip(".")
    if ext in ("mkv", "mka", "flac"):
        return ["-c:a", "flac", "-sample_fmt", "s32"]
    if ext in ("mov", "alac"):
        return ["-c:a", "alac", "-sample_fmt", "s32p"]
    if ext in ("wav",):
        return ["-c:a", "pcm_s24le"]
    br = 320 if channels <= 2 else 160 * channels                     # MP4 / M4A: high-bitrate AAC
    return ["-c:a", "aac", "-b:a", f"{br}k"]


AUDIO_FORMATS = {   # label -> (extension, encoder args)
    "FLAC (lossless)": ("flac", ["-c:a", "flac", "-sample_fmt", "s32"]),
    "WAV (24-bit)": ("wav", ["-c:a", "pcm_s24le"]),
    "M4A (ALAC, lossless)": ("m4a", ["-c:a", "alac", "-sample_fmt", "s32p"]),
    "M4A (AAC, 320 kbps)": ("m4a", ["-c:a", "aac", "-b:a", "320k"]),
}


class NotNeeded(Exception):
    """The audio is already full range: restoring would change nothing."""


def restore_audio_file(input_path, output_path, gpu_ids, steps=audio.DEFAULT_STEPS, on_log=print,
                       on_progress=None, cancel_event=None, codec_args=None):
    """Audio file in -> restored audio file out. Resumable. Raises NotNeeded for full-range audio."""
    work = output_path + ".audio_parts"
    job = AudioRestoreJob(input_path, work, gpu_ids, steps, on_log=on_log, on_progress=on_progress,
                          cancel_event=cancel_event)
    try:
        wav = job.run()
    finally:
        audio.release()
    if wav is False:
        return False
    if wav is None:
        shutil.rmtree(work, ignore_errors=True)
        raise NotNeeded(f"already full range ({job.cutoff / 1000:.0f} kHz)")
    info = probe_audio(input_path)
    cmd = [find_tool("ffmpeg"), "-v", "error", "-y", "-i", wav, "-i", input_path, "-map", "0:a:0",
           "-map_metadata", "1",
           *(codec_args or audio_codec_args(os.path.splitext(output_path)[1], info.channels))]
    tmp = output_path + ".writing" + os.path.splitext(output_path)[1]
    r = subprocess.run(cmd + [tmp], capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Encoding the audio failed: " + r.stderr.strip()[-300:])
    os.replace(tmp, output_path)
    shutil.rmtree(work, ignore_errors=True)
    return job.cutoff


def mux_audio_into_video(video_path, audio_wav, output_path, info_channels):
    """Copy the video stream untouched and replace the first audio track with the restored one."""
    ext = os.path.splitext(output_path)[1]
    cmd = [find_tool("ffmpeg"), "-v", "error", "-y", "-i", video_path, "-i", audio_wav,
           "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
           *audio_codec_args(ext, info_channels)]
    if ext.lower() in (".mp4", ".m4v", ".mov"):
        cmd += ["-movflags", "+faststart"]
    tmp = output_path + ".writing" + ext
    r = subprocess.run(cmd + [tmp], capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Writing the video with restored audio failed: " + r.stderr.strip()[-300:])
    os.replace(tmp, output_path)
