"""Build MUSS.exe (launcher) and MUSS-Setup.exe (installer) with PyInstaller.

    python installer/build.py --version v0.3.0        ->  dist/MUSS-Setup.exe

Used by .github/workflows/release.yml on every v* tag; also works locally.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build")
DIST = os.path.join(ROOT, "dist")


def pyinstaller(*args):
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
           "--workpath", os.path.join(BUILD, "work"), "--specpath", os.path.join(BUILD, "spec"), *args]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="main", help="git tag the installer installs, e.g. v0.3.0")
    a = ap.parse_args()
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    with open(os.path.join(HERE, "_build_info.py"), "w", encoding="utf-8") as f:
        f.write(f'VERSION = "{a.version}"\n')
    launcher_dir = os.path.join(BUILD, "launcher")
    pyinstaller("--name", "MUSS", "--distpath", launcher_dir, os.path.join(HERE, "muss_launcher.py"))
    pyinstaller("--name", "MUSS-Setup", "--distpath", DIST, "--paths", HERE,
                "--add-data", f"{os.path.join(launcher_dir, 'MUSS.exe')}{os.pathsep}.",
                "--add-data", f"{os.path.join(HERE, 'banner.png')}{os.pathsep}.",
                "--hidden-import", "_build_info", os.path.join(HERE, "muss_setup.py"))
    out = os.path.join(DIST, "MUSS-Setup.exe")
    print(f"built {out} ({os.path.getsize(out) / 1e6:.1f} MB) for {a.version}")


if __name__ == "__main__":
    main()
