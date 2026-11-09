"""Publish a new MUSS version (for the developer).

    .venv\\Scripts\\python.exe release.py 0.3.0 "Short summary of the release"

Sets the version in upscaler/__init__.py, commits it, creates the tag v0.3.0 and pushes the
commit and tag to GitHub. Every MUSS installed with git then offers this update at its next start.
Refuses to run with uncommitted changes, so a release always matches what is committed.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INIT = os.path.join(ROOT, "upscaler", "__init__.py")


def git(*args):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return r.stdout.strip()


def main():
    if len(sys.argv) < 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        sys.exit(__doc__)
    version = sys.argv[1]
    summary = sys.argv[2] if len(sys.argv) > 2 else f"MUSS {version}"
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("Commit or discard your changes first.")
    tag = f"v{version}"
    if tag in git("tag", "-l", tag).split():
        sys.exit(f"{tag} already exists.")
    with open(INIT, encoding="utf-8") as f:
        text = f.read()
    current = re.search(r'__version__ = "([^"]+)"', text).group(1)
    if current != version:
        with open(INIT, "w", encoding="utf-8", newline="\n") as f:
            f.write(re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', text))
        git("commit", "-q", "-am", f"Release {tag}: {summary}")
    git("tag", "-a", tag, "-m", summary)
    git("push", "origin", "HEAD")
    git("push", "origin", tag)
    print(f"Released {tag}. Installed copies will offer the update at their next start.")


if __name__ == "__main__":
    main()
