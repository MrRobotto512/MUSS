"""Self-update with git.

MUSS is installed as a git clone of the GitHub repo. A release is a version tag (v0.3.0, ...) on
GitHub; pushing commits to main without a tag never updates anyone.

    check  : git fetch --tags, find the highest v* tag newer than this version
    notes  : the commit messages between this version and that tag
    apply  : fast-forward to the tag (refused if there are uncommitted changes or the branch
             has diverged), reinstall Python packages only if requirements.txt changed,
             roll back with git if that fails, then restart
"""
import os
import shutil
import subprocess
import sys

from . import GITHUB_REPO, __version__

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def parse_version(v: str):
    v = v.strip().lstrip("vV")
    parts = []
    for p in v.split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def app_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UpdateError(RuntimeError):
    pass


def _git(root, *args, check=True, timeout=120):
    exe = shutil.which("git")
    if not exe:
        raise UpdateError("Git is not installed. Install Git for Windows (https://git-scm.com) to get updates.")
    r = subprocess.run([exe, "-C", root, *args], capture_output=True, text=True, timeout=timeout,
                       creationflags=NO_WINDOW)
    if check and r.returncode != 0:
        raise UpdateError(f"git {' '.join(args)} failed: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout.strip()


def status(root=None) -> str | None:
    """None if updates can work here, otherwise a human-readable reason."""
    root = root or app_dir()
    if not shutil.which("git"):
        return "Git is not installed, so MUSS can't update itself."
    if not os.path.isdir(os.path.join(root, ".git")):
        return (f"This copy of MUSS wasn't installed with git. To get updates, install it with:\n"
                f"git clone https://github.com/{GITHUB_REPO}.git")
    return None


def check_latest(root=None):
    """Fetch tags; return info about the newest release tag newer than this version, else None."""
    root = root or app_dir()
    reason = status(root)
    if reason:
        raise UpdateError(reason)
    _git(root, "fetch", "--tags", "--force", "--quiet", "origin", timeout=60)
    tags = [t for t in _git(root, "tag", "-l", "v*").splitlines() if t.strip()]
    if not tags:
        return None
    latest = max(tags, key=parse_version)
    version = latest.lstrip("vV")
    if not is_newer(version):
        return None
    log = _git(root, "log", "--no-merges", "--format=- %s", f"HEAD..{latest}", check=False)
    return {"version": version, "tag": latest, "notes": log or "(no change notes)",
            "url": f"https://github.com/{GITHUB_REPO}/releases/tag/{latest}"}


def apply_update(release, root=None, log=print, run_pip=True):
    """Fast-forward to the release tag. Returns the new version; rolls back on failure."""
    root = root or app_dir()
    reason = status(root)
    if reason:
        raise UpdateError(reason)
    dirty = [l for l in _git(root, "status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    if dirty:
        raise UpdateError("You have uncommitted changes in the MUSS folder, so the update was not applied "
                          "(it would overwrite them):\n" + "\n".join(dirty[:10]))
    tag = release["tag"]
    old = _git(root, "rev-parse", "HEAD")
    is_ancestor = subprocess.run([shutil.which("git"), "-C", root, "merge-base", "--is-ancestor", "HEAD", tag],
                                 capture_output=True, creationflags=NO_WINDOW).returncode == 0
    if not is_ancestor:
        raise UpdateError(f"This copy has commits that aren't in {tag}, so it can't be fast-forwarded. "
                          "Update it with git manually.")
    req_before = _git(root, "rev-parse", "HEAD:requirements.txt", check=False)
    req_after = _git(root, "rev-parse", f"{tag}:requirements.txt", check=False)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    log(f"Updating MUSS {__version__} -> {release['version']}...")
    if branch == "HEAD":                                  # detached (e.g. a previous tag): just move
        _git(root, "checkout", "--quiet", tag)
    else:
        _git(root, "merge", "--ff-only", "--quiet", tag)
    try:
        if run_pip and req_before != req_after:
            log("requirements.txt changed - updating Python packages (this can take a while)...")
            py = sys.executable.replace("pythonw.exe", "python.exe")
            r = subprocess.run([py, "-m", "pip", "install", "-r", os.path.join(root, "requirements.txt"),
                                "--extra-index-url", "https://download.pytorch.org/whl/cu128"],
                               capture_output=True, text=True, creationflags=NO_WINDOW, cwd=root)
            if r.returncode != 0:
                raise UpdateError("Installing Python packages failed:\n" + (r.stderr or r.stdout)[-800:])
    except Exception:
        log("Update failed - going back to the previous version...")
        _git(root, "reset", "--hard", "--quiet", old)
        raise
    log(f"Updated to MUSS {release['version']}.")
    return release["version"]


def restart_app(root=None):
    """Start a fresh MUSS process (the caller then quits)."""
    root = root or app_dir()
    exe = sys.executable
    if exe.lower().endswith("python.exe") and os.path.exists(exe[:-10] + "pythonw.exe"):
        exe = exe[:-10] + "pythonw.exe"
    subprocess.Popen([exe, os.path.join(root, "app.py")], cwd=root,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
