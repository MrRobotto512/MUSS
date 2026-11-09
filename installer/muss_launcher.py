r"""MUSS.exe - starts MUSS from an installation made by MUSS-Setup.exe.

Layout next to this exe:
    app\            the git checkout (with app\.venv)
    tools\mingit\   portable Git (only if Git wasn't installed)
    tools\ffmpeg\   ffmpeg with NVENC (only if none was found)
"""
import ctypes
import os
import subprocess
import sys


def main():
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    app = os.path.join(base, "app")
    pyw = os.path.join(app, ".venv", "Scripts", "pythonw.exe")
    if not (os.path.isfile(pyw) and os.path.isfile(os.path.join(app, "app.py"))):
        ctypes.windll.user32.MessageBoxW(None, "MUSS isn't installed correctly.\n\nRun MUSS-Setup.exe again to repair it.",
                                         "MUSS", 0x10)
        return 1
    env = dict(os.environ)
    extra = [os.path.join(base, "tools", "mingit", "cmd"), os.path.join(base, "tools", "ffmpeg", "bin")]
    env["PATH"] = os.pathsep.join([p for p in extra if os.path.isdir(p)] + [env.get("PATH", "")])
    env["MUSS_HOME"] = base
    subprocess.Popen([pyw, os.path.join(app, "app.py"), *sys.argv[1:]], cwd=app, env=env,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
