"""Rendering on several PCs.

Render node (the other PC): `NodeServer` listens on the local network (default port 47820).
Every request must carry the node's pairing code. For each export it downloads the source video
(and a custom model if needed) from the main PC once, then renders segments on request with the
same pipeline as local GPUs (NVDEC -> AI -> NVENC) and returns each finished segment file.

Main PC: one `RemoteWorker` per GPU of each node takes segments from the shared queue, exactly
like a local GPU. A tiny token-protected file server lets nodes download the source and model.
"""
import hashlib
import http.server
import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
import urllib.parse
import urllib.request

DEFAULT_PORT = 47820
PROTOCOL = 1
_CHUNK = 1 << 20


def lan_ip() -> str:
    """This PC's address on the local network (the one used for the default route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 9))          # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def new_pairing_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


# =========================================================================== main PC: file server
class FileServer:
    """Serves registered files (source video, custom model) to render nodes, token-protected."""

    def __init__(self):
        self.token = secrets.token_urlsafe(24)
        self.files: dict[str, str] = {}
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                q = urllib.parse.urlparse(self.path)
                fid = q.path.strip("/")
                if self.headers.get("X-Token") != outer.token or fid not in outer.files:
                    self.send_error(403)
                    return
                path = outer.files[fid]
                size = os.path.getsize(path)
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, _CHUNK)

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def register(self, path: str) -> str:
        fid = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:20]
        self.files[fid] = path
        return fid


_FILE_SERVER = None
_FS_LOCK = threading.Lock()


def file_server() -> FileServer:
    global _FILE_SERVER
    with _FS_LOCK:
        if _FILE_SERVER is None:
            _FILE_SERVER = FileServer()
        return _FILE_SERVER


# =========================================================================== main PC: client
def _request(host, port, code, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://{host}:{port}{path}", data=data, method="POST" if data else "GET",
                                 headers={"X-Pairing-Code": code, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def node_info(host, port, code, timeout=4) -> dict:
    with _request(host, port, code, "/info", timeout=timeout) as r:
        return json.loads(r.read())


class RemoteWorker:
    """One GPU on a render node, used by ExportJob like a local GPU."""

    def __init__(self, host, port, code, gpu_index, label):
        self.host, self.port, self.code, self.gpu = host, int(port), code, gpu_index
        self.name = label

    def prepare(self, job, info):
        fs = file_server()
        # the address of this PC as seen from the node
        with socket.create_connection((self.host, self.port), timeout=5) as s:
            my_ip = s.getsockname()[0]
        spec = job.spec
        payload = {
            "protocol": PROTOCOL, "job_id": self.job_id(job), "source_name": os.path.basename(job.input_path),
            "source_size": os.path.getsize(job.input_path),
            "source_url": f"http://{my_ip}:{fs.port}/{fs.register(job.input_path)}", "file_token": fs.token,
            "model": ({"builtin": spec.key} if spec.builtin else
                      {"custom": os.path.basename(spec.path), "size": os.path.getsize(spec.path),
                       "url": f"http://{my_ip}:{fs.port}/{fs.register(spec.path)}"}),
            "encoder": job.encoder_label, "cq": job.cq, "use_tensorrt": job.use_tensorrt,
        }
        job.log(f"{self.name}: preparing (copying the video to the node, loading the model)...")
        with _request(self.host, self.port, self.code, "/prepare", payload, timeout=3600) as r:
            res = json.loads(r.read())
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "prepare failed"))
        job.log(f"{self.name}: ready ({res.get('decoder', '')}, {res.get('backend', '')}).")

    @staticmethod
    def job_id(job):
        st = os.stat(job.input_path)
        key = f"{os.path.abspath(job.input_path)}|{st.st_size}|{int(st.st_mtime)}|{job.spec.key}|{job.encoder_label}|{job.cq}|{job.use_tensorrt}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def render_segment(self, job, seg, dest_path, frame_done):
        i, start, count = seg
        payload = {"job_id": self.job_id(job), "gpu": self.gpu, "index": i, "start": start, "count": count}
        tmp = dest_path + ".tmp"
        with _request(self.host, self.port, self.code, "/render", payload, timeout=3600) as r, open(tmp, "wb") as f:
            expected = int(r.headers.get("Content-Length") or -1)
            got = 0
            while chunk := r.read(_CHUNK):
                if job.cancel_event.is_set():
                    raise RuntimeError("cancelled")
                f.write(chunk)
                got += len(chunk)
        if expected >= 0 and got != expected:
            raise RuntimeError(f"segment {i} arrived incomplete ({got}/{expected} bytes)")
        os.replace(tmp, dest_path)
        for _ in range(count):
            frame_done()

    def cancel(self, job):
        try:
            with _request(self.host, self.port, self.code, "/cancel", {"job_id": self.job_id(job)}, timeout=3):
                pass
        except OSError:
            pass


def remote_workers_for(nodes, log=print):
    """[{host, port, code, name}] -> RemoteWorker per GPU of every reachable node."""
    workers = []
    for n in nodes:
        try:
            info = node_info(n["host"], n.get("port", DEFAULT_PORT), n["code"])
        except Exception as e:  # noqa: BLE001
            log(f"Render node {n.get('name') or n['host']} is not reachable ({e}); skipping it.")
            continue
        for g in info.get("gpus", []):
            workers.append(RemoteWorker(n["host"], n.get("port", DEFAULT_PORT), n["code"], g["index"],
                                        f"{info.get('name', n['host'])} GPU {g['index']}"))
    return workers


# =========================================================================== render node: server
class NodeServer:
    """Runs on the helper PC. Start with NodeServer(...).start(); stop with .stop()."""

    def __init__(self, port=DEFAULT_PORT, code=None, log=print, gpu_ids=None):
        self.port, self.code, self.log = port, code or new_pairing_code(), log
        self.workdir = os.path.join(tempfile.gettempdir(), "ai-upscaler-node")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.httpd = None
        self.gpu_ids = gpu_ids
        self.busy = 0

    # ---- lifecycle
    def start(self):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _auth(self):
                if not secrets.compare_digest(self.headers.get("X-Pairing-Code", ""), outer.code):
                    self._json(403, {"ok": False, "error": "wrong pairing code"})
                    return False
                return True

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):
                if not self._auth():
                    return
                if self.path == "/info":
                    self._json(200, outer.info())
                else:
                    self._json(404, {"ok": False})

            def do_POST(self):
                if not self._auth():
                    return
                try:
                    body = self._body()
                    if self.path == "/prepare":
                        self._json(200, outer.prepare(body))
                    elif self.path == "/cancel":
                        outer.cancel(body.get("job_id"))
                        self._json(200, {"ok": True})
                    elif self.path == "/render":
                        path = outer.render(body)
                        size = os.path.getsize(path)
                        self.send_response(200)
                        self.send_header("Content-Type", "video/x-matroska")
                        self.send_header("Content-Length", str(size))
                        self.end_headers()
                        with open(path, "rb") as f:
                            shutil.copyfileobj(f, self.wfile, _CHUNK)
                        os.remove(path)
                    else:
                        self._json(404, {"ok": False})
                except Exception as e:  # noqa: BLE001
                    outer.log(f"Request {self.path} failed: {e}")
                    self._json(500, {"ok": False, "error": str(e)})

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.log(f"Render node listening on {lan_ip()}:{self.port}  -  pairing code {self.code}")
        return self

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        for j in list(self.jobs.values()):
            j["job"].cancel()
        self.log("Render node stopped.")

    # ---- API
    def info(self):
        from . import engine, gpu_io, trt_backend
        gpus = [{"index": i, "name": n} for i, n in engine.list_gpus() if self.gpu_ids is None or i in self.gpu_ids]
        return {"ok": True, "protocol": PROTOCOL, "name": socket.gethostname(), "gpus": gpus, "busy": self.busy,
                "nvdec": gpu_io.nvdec_available(), "tensorrt": trt_backend.status()}

    def prepare(self, req):
        from . import engine, export, models
        if req.get("protocol") != PROTOCOL:
            return {"ok": False, "error": "this render node runs a different app version"}
        jid = req["job_id"]
        with self.lock:
            entry = self.jobs.get(jid)
            if entry is None:
                entry = self.jobs[jid] = {"lock": threading.Lock(), "ready": False}
        with entry["lock"]:                       # several GPUs of this node may ask at once
            if entry["ready"]:
                entry["job"].cancel_event.clear()
                return {"ok": True, "decoder": entry["decoder"], "backend": entry["backend"]}
            d = os.path.join(self.workdir, jid)
            os.makedirs(d, exist_ok=True)
            src = os.path.join(d, "source" + os.path.splitext(req["source_name"])[1])
            if not (os.path.exists(src) and os.path.getsize(src) == req["source_size"]):
                self.log(f"Downloading {req['source_name']} ({req['source_size'] / 1e6:.0f} MB) from the main PC...")
                _download(req["source_url"], req["file_token"], src)
            m = req["model"]
            if "builtin" in m:
                spec = models.MODELS_BY_KEY[m["builtin"]]
            else:
                mp = os.path.join(d, m["custom"])
                if not (os.path.exists(mp) and os.path.getsize(mp) == m["size"]):
                    _download(m["url"], req["file_token"], mp)
                spec = models._spec_from_meta(mp, models.inspect_model_file(mp))
            info = engine.probe(src)
            gpus = [i for i, _ in engine.list_gpus() if self.gpu_ids is None or i in self.gpu_ids]
            job = export.ExportJob(src, os.path.join(d, "node_output.mkv"), spec, gpus, req["encoder"], cq=req["cq"],
                                   use_tensorrt=req["use_tensorrt"], on_log=self.log)
            job.info, job.workdir = info, d
            nets = engine.get_nets(spec, gpus, req["use_tensorrt"], info.width, info.height, log=self.log)
            readers = {}
            for g in gpus:
                readers[g] = engine.gpu_io.open_reader(src, info, g, engine.find_tool("ffmpeg"),
                                                       engine._in_matrix(info), log=self.log)
            backend = "TensorRT" if any(getattr(n, "is_tensorrt", False) for n in nets.values()) else "PyTorch"
            entry.update(job=job, nets=nets, readers=readers, info=info, ready=True,
                         decoder=next(iter(readers.values())).kind, backend=backend,
                         gpu_locks={g: threading.Lock() for g in gpus})
            self.log(f"Ready for {req['source_name']}: {spec.label}, {backend}, decoding on {entry['decoder']}.")
            return {"ok": True, "decoder": entry["decoder"], "backend": backend}

    def render(self, req):
        entry = self.jobs.get(req["job_id"])
        if not entry or not entry.get("ready"):
            raise RuntimeError("job not prepared")
        g = int(req["gpu"])
        job = entry["job"]
        with entry["gpu_locks"][g]:
            self.busy += 1
            try:
                t0 = time.time()
                job._render_segment(g, entry["nets"][g], entry["readers"][g], entry["info"],
                                    (req["index"], req["start"], req["count"]), lambda: None)
                path = job._seg_path(req["index"])
                if not os.path.exists(path):
                    raise RuntimeError("segment was cancelled")
                self.log(f"GPU {g}: segment {req['index']} ({req['count']} frames) in {time.time() - t0:.1f}s")
                return path
            finally:
                self.busy -= 1

    def cancel(self, job_id):
        entry = self.jobs.get(job_id)
        if entry and entry.get("job"):
            entry["job"].cancel()


def _download(url, token, dest):
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"X-Token": token})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, _CHUNK)
    os.replace(tmp, dest)
