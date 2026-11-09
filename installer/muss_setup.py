r"""MUSS-Setup.exe - installs MUSS on Windows without needing Python, Git or ffmpeg beforehand.

Everything goes into one folder (default %LOCALAPPDATA%\Programs\MUSS, no admin rights needed):
    app\                 git checkout of MUSS at the release tag (so MUSS can update itself)
    app\.venv\           Python 3.14 environment with CUDA PyTorch, TensorRT, ... (several GB)
    tools\uv\            uv (Astral) - fetches Python and installs packages
    tools\python\        the Python 3.14 used by the environment
    tools\mingit\        portable Git, only if Git isn't installed
    tools\ffmpeg\        ffmpeg with NVENC/NVDEC, only if none is found
    MUSS.exe             launcher;  uninstall.cmd  removes everything

Downloads come from their official GitHub releases and are checked against the published
SHA-256 checksums where the project publishes them (uv, ffmpeg builds, MinGit).

Command line (also used for automated tests):
    MUSS-Setup.exe [--dir PATH] [--yes] [--no-shortcuts] [--no-register] [--repo URL] [--ref TAG]
"""
import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile

try:
    from _build_info import VERSION          # written by installer/build.py, e.g. "v0.3.0"
except ImportError:
    VERSION = "main"

APP = "MUSS"
REPO = "https://github.com/MrRobotto512/MUSS.git"
PY_VERSION = "3.14"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
UA = {"User-Agent": f"MUSS-Setup/{VERSION}"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DEFAULT_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Programs", APP)


class SetupError(Exception):
    pass


# =========================================================================== helpers
def _get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def download(url, dest, log, progress=None):
    """Download to dest; returns the SHA-256 of the file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = hashlib.sha256()
    tmp = dest + ".part"
    with _get(url, 120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total)
    os.replace(tmp, dest)
    return h.hexdigest()


def check(actual, expected, what, log=print):
    if not expected:
        log(f"  {what}: no published checksum found - downloaded over HTTPS from the official release")
        return
    if actual.lower() != expected.lower():
        raise SetupError(f"{what}: checksum mismatch - the download is corrupt or was tampered with.")
    log(f"  {what}: SHA-256 verified")


def unzip(path, dest):
    root = os.path.realpath(dest)
    with zipfile.ZipFile(path) as z:
        for m in z.infolist():
            t = os.path.realpath(os.path.join(dest, m.filename))
            if not (t == root or t.startswith(root + os.sep)):
                raise SetupError(f"Unsafe path in {os.path.basename(path)}")
        z.extractall(dest)


def run(cmd, log, env=None, cwd=None):
    log("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd,
                         creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-15:]
            log("  " + line)
    if p.wait() != 0:
        raise SetupError(f"{os.path.basename(cmd[0])} failed:\n" + "\n".join(tail[-6:]))


def gh_release(repo, tag="latest"):
    url = f"https://api.github.com/repos/{repo}/releases/" + ("latest" if tag == "latest" else f"tags/{tag}")
    with _get(url) as r:
        return json.loads(r.read())


def have_nvenc_ffmpeg():
    topaz = r"C:\Program Files\Topaz Video AI\ffmpeg.exe"
    exe = topaz if os.path.exists(topaz) else shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20,
                             creationflags=NO_WINDOW).stdout
        return "hevc_nvenc" in out
    except (OSError, subprocess.SubprocessError):
        return False


# =========================================================================== install steps
class Installer:
    def __init__(self, base, log, progress, shortcuts=True, register=True, repo=REPO, ref=VERSION):
        self.base, self.progress = os.path.abspath(base), progress
        os.makedirs(self.base, exist_ok=True)
        self._logfile = open(os.path.join(self.base, "setup.log"), "a", encoding="utf-8")

        def log_both(m):
            self._logfile.write(m + "\n")
            self._logfile.flush()
            log(m)
        self.log = log_both
        self.shortcuts, self.register, self.repo, self.ref = shortcuts, register, repo, ref
        self.tools = os.path.join(self.base, "tools")
        self.app = os.path.join(self.base, "app")
        self.git = None

    def env(self):
        e = dict(os.environ)
        e["UV_PYTHON_INSTALL_DIR"] = os.path.join(self.tools, "python")
        # short path on purpose: some packages (tensorrt-cu12) build from source, and deep
        # cache paths exceed Windows' 260-character path limit during that build
        e["UV_CACHE_DIR"] = os.path.join(os.environ.get("TEMP", self.tools), "muss-uv")
        if self.git and os.path.dirname(self.git) not in e.get("PATH", ""):
            e["PATH"] = os.path.dirname(self.git) + os.pathsep + e.get("PATH", "")
        return e

    def steps(self):
        return [("Checking the GPU", self.check_gpu), ("Getting Git", self.get_git), ("Getting uv", self.get_uv),
                ("Getting ffmpeg", self.get_ffmpeg), ("Downloading MUSS", self.get_app),
                ("Installing Python and packages (several GB)", self.get_packages),
                ("Checking the installation", self.verify), ("Creating shortcuts", self.finish)]

    def check_gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if not out:
            self.log("WARNING: no NVIDIA GPU / driver found. MUSS needs an NVIDIA RTX GPU to upscale.")
        for line in out.splitlines():
            self.log(f"  GPU: {line}")

    def get_git(self):
        self.git = shutil.which("git")
        mingit = os.path.join(self.tools, "mingit", "cmd", "git.exe")
        if self.git:
            self.log(f"  using installed Git: {self.git}")
            return
        if os.path.exists(mingit):
            self.git = mingit
            return
        rel = gh_release("git-for-windows/git")
        asset = next(a for a in rel["assets"] if re.fullmatch(r"MinGit-[\d.]+(\.windows\.\d+)?-64-bit\.zip", a["name"]))
        m = re.search(re.escape(asset["name"]) + r"\s*\|\s*([0-9a-f]{64})", rel.get("body", ""))
        self.log(f"  downloading {asset['name']} (portable Git, {asset['size'] / 1e6:.0f} MB)")
        z = os.path.join(self.tools, "mingit.zip")
        check(download(asset["browser_download_url"], z, self.log, self.progress), m.group(1) if m else None, "MinGit", self.log)
        unzip(z, os.path.join(self.tools, "mingit"))
        os.remove(z)
        self.git = mingit

    def get_uv(self):
        uv = os.path.join(self.tools, "uv", "uv.exe")
        if os.path.exists(uv):
            self.uv = uv
            return
        base = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        with _get(base + ".sha256") as r:
            expected = r.read().decode().split()[0]
        z = os.path.join(self.tools, "uv.zip")
        self.log("  downloading uv")
        check(download(base, z, self.log, self.progress), expected, "uv", self.log)
        unzip(z, os.path.join(self.tools, "uv"))
        os.remove(z)
        self.uv = uv

    def get_ffmpeg(self):
        dest = os.path.join(self.tools, "ffmpeg")
        if os.path.exists(os.path.join(dest, "bin", "ffmpeg.exe")):
            return
        if have_nvenc_ffmpeg():
            self.log("  an ffmpeg with NVENC is already available - not downloading one")
            return
        rel = gh_release("BtbN/FFmpeg-Builds", "latest")
        name = "ffmpeg-master-latest-win64-lgpl.zip"
        asset = next(a for a in rel["assets"] if a["name"] == name)
        sums = next(a for a in rel["assets"] if a["name"] == "checksums.sha256")
        with _get(sums["browser_download_url"]) as r:
            m = re.search(r"([0-9a-f]{64})\s+\*?" + re.escape(name), r.read().decode())
        self.log(f"  downloading {name} ({asset['size'] / 1e6:.0f} MB)")
        z = os.path.join(self.tools, "ffmpeg.zip")
        check(download(asset["browser_download_url"], z, self.log, self.progress), m.group(1) if m else None, "ffmpeg", self.log)
        tmp = os.path.join(self.tools, "ffmpeg_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        unzip(z, tmp)
        top = os.path.join(tmp, os.listdir(tmp)[0])
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(top, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        os.remove(z)

    def get_app(self):
        if os.path.isdir(os.path.join(self.app, ".git")):
            self.log("  existing installation found - updating it")
            run([self.git, "-C", self.app, "fetch", "--tags", "--force", "origin"], self.log, self.env())
            run([self.git, "-C", self.app, "checkout", "--force", self.ref], self.log, self.env())
        else:
            if os.path.exists(self.app):
                shutil.rmtree(self.app)
            run([self.git, "clone", "--branch", self.ref, self.repo, self.app], self.log, self.env())

    def get_packages(self):
        venv = os.path.join(self.app, ".venv")
        py = os.path.join(venv, "Scripts", "python.exe")
        if not os.path.exists(py):
            # --seed: include pip, which MUSS's self-update uses when requirements change
            run([self.uv, "venv", "--seed", "--python", PY_VERSION, venv], self.log, self.env())
        run([self.uv, "pip", "install", "--python", py, "-r", os.path.join(self.app, "requirements.txt"),
             "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"], self.log, self.env())
        run([self.uv, "cache", "clean"], self.log, self.env())         # frees several GB

    def verify(self):
        py = os.path.join(self.app, ".venv", "Scripts", "python.exe")
        code = ("import torch, PySide6, tensorrt, spandrel; "
                "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', "
                "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])")
        run([py, "-c", code], self.log, self.env())

    def finish(self):
        launcher_src = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "MUSS.exe")
        exe = os.path.join(self.base, "MUSS.exe")
        if os.path.exists(launcher_src):
            shutil.copy2(launcher_src, exe)
        else:
            self.log("  (running from source: no MUSS.exe launcher to copy)")
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        start_menu = os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs")
        links = [os.path.join(start_menu, "MUSS.lnk"), os.path.join(desktop, "MUSS.lnk")] if self.shortcuts else []
        for link in links:
            ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
                  f"$s.TargetPath='{exe}';$s.WorkingDirectory='{self.base}';"
                  f"$s.Description='MUSS - Movie UpScaling Software';$s.Save()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], creationflags=NO_WINDOW)
        with open(os.path.join(self.base, "uninstall.cmd"), "w", encoding="utf-8") as f:
            f.write("@echo off\r\necho Uninstalling MUSS...\r\n")
            for link in links:
                f.write(f'del "{link}" 2>nul\r\n')
            f.write('reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MUSS" /f >nul 2>&1\r\n')
            f.write(f'cd /d "%TEMP%"\r\nstart "" /b cmd /c "timeout /t 2 >nul & rmdir /s /q ""{self.base}"""\r\n')
        if self.register:
            import winreg
            k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall\MUSS")
            for name, val in (("DisplayName", "MUSS - Movie UpScaling Software"), ("DisplayVersion", self.ref.lstrip("v")),
                              ("Publisher", "MUSS contributors"), ("InstallLocation", self.base),
                              ("DisplayIcon", exe), ("UninstallString", f'"{os.path.join(self.base, "uninstall.cmd")}"'),
                              ("URLInfoAbout", "https://github.com/MrRobotto512/MUSS")):
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, val)
            winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.CloseKey(k)

    def install(self):
        os.makedirs(self.tools, exist_ok=True)
        for title, fn in self.steps():
            self.log(f"== {title}")
            self.progress(None)
            fn()
        self.log("== MUSS is installed.")


# =========================================================================== UI
def gui(args):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(f"MUSS Setup ({VERSION})")
    root.geometry("760x520")
    frm = ttk.Frame(root, padding=14)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text="MUSS - Movie UpScaling Software", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(frm, text="Installs MUSS with everything it needs. Downloads several GB (mostly CUDA PyTorch); "
                        "an NVIDIA RTX GPU is required.", wraplength=720).pack(anchor="w", pady=(4, 10))
    row = ttk.Frame(frm)
    row.pack(fill="x")
    ttk.Label(row, text="Install to").pack(side="left")
    dir_var = tk.StringVar(value=args.dir or DEFAULT_DIR)
    ttk.Entry(row, textvariable=dir_var).pack(side="left", fill="x", expand=True, padx=8)
    ttk.Button(row, text="Browse...", command=lambda: dir_var.set(
        filedialog.askdirectory(initialdir=dir_var.get()) or dir_var.get())).pack(side="left")
    sc_var = tk.BooleanVar(value=not args.no_shortcuts)
    ttk.Checkbutton(frm, text="Create Start menu and Desktop shortcuts", variable=sc_var).pack(anchor="w", pady=6)
    bar = ttk.Progressbar(frm, mode="indeterminate")
    bar.pack(fill="x", pady=(4, 6))
    logbox = tk.Text(frm, height=14, font=("Consolas", 9), state="disabled")
    logbox.pack(fill="both", expand=True)
    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(8, 0))
    go = ttk.Button(btns, text="Install")
    go.pack(side="right")
    launch = ttk.Button(btns, text="Launch MUSS")
    events = queue.Queue()

    def pump():
        try:
            while True:
                kind, val = events.get_nowait()
                if kind == "log":
                    logbox.configure(state="normal")
                    logbox.insert("end", val + "\n")
                    logbox.see("end")
                    logbox.configure(state="disabled")
                elif kind == "progress":
                    if val is None:
                        bar.configure(mode="indeterminate")
                        bar.start(12)
                    else:
                        bar.stop()
                        bar.configure(mode="determinate", value=val * 100)
                elif kind == "done":
                    bar.stop()
                    bar.configure(mode="determinate", value=100 if val is None else 0)
                    if val is None:
                        go.configure(text="Close", state="normal", command=root.destroy)
                        launch.configure(command=lambda: (subprocess.Popen([os.path.join(dir_var.get(), "MUSS.exe")]),
                                                          root.destroy()))
                        launch.pack(side="right", padx=8)
                        messagebox.showinfo("MUSS Setup", "MUSS is installed.")
                    else:
                        go.configure(text="Retry", state="normal")
                        messagebox.showerror("MUSS Setup", f"Installation failed:\n\n{val}")
        except queue.Empty:
            pass
        root.after(100, pump)

    def start():
        go.configure(state="disabled")
        inst = Installer(dir_var.get(), lambda m: events.put(("log", m)), lambda v: events.put(("progress", v)),
                         shortcuts=sc_var.get(), register=not args.no_register, repo=args.repo, ref=args.ref)

        def work():
            try:
                inst.install()
                events.put(("done", None))
            except Exception as e:  # noqa: BLE001
                events.put(("log", f"ERROR: {e}"))
                events.put(("done", str(e)))
        threading.Thread(target=work, daemon=True).start()

    go.configure(command=start)
    root.after(100, pump)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Install MUSS")
    ap.add_argument("--dir")
    ap.add_argument("--yes", action="store_true", help="install without the window (unattended)")
    ap.add_argument("--no-shortcuts", action="store_true")
    ap.add_argument("--no-register", action="store_true", help="don't add an entry to Windows' installed apps")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--ref", default=VERSION)
    args = ap.parse_args()
    if not args.yes:
        return gui(args)
    inst = Installer(args.dir or DEFAULT_DIR, lambda m: print(m, flush=True), lambda v: None,
                     shortcuts=not args.no_shortcuts, register=not args.no_register, repo=args.repo, ref=args.ref)
    try:
        inst.install()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
