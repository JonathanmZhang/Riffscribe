"""Which code a worker image was built from, and a guard against measuring
with a stale image.

At build time (worker/Dockerfile, last layers) `python -m scripts.build_info
record` writes /app/BUILD_INFO.json: the git commit the image was built
from (resolved from the repo's .git, passed in as the "gitmeta" build
context by docker-compose.yml) and a hash of the worker code in the image
(tasks/, scripts/, requirements*.txt).

At run time, guard() compares that with the checkout mounted read-only at
/repo (docker-compose.yml mounts it into both worker services; with a plain
`docker run`, add -v <repo>:/repo:ro). It prints both at the top of the
report and exits if the commit or the worker code differs - an old image
measures old code. --allow-stale runs anyway, with a warning.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

BUILD_INFO = "/app/BUILD_INFO.json"
IMAGE_CODE = "/app"
REPO = os.environ.get("REPO_DIR", "/repo")
BUILD_GIT_DIR = "/tmp/gitmeta"


def resolve_head(git_dir: str) -> str | None:
    """HEAD's commit from a .git directory (loose ref or packed-refs), or
    None if it can't be read."""
    try:
        head = open(os.path.join(git_dir, "HEAD")).read().strip()
    except OSError:
        return None
    if not head.startswith("ref: "):
        return head  # detached HEAD
    ref = head[5:]
    try:
        return open(os.path.join(git_dir, ref)).read().strip()
    except OSError:
        pass
    try:
        for line in open(os.path.join(git_dir, "packed-refs")):
            parts = line.strip().split(" ")
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        pass
    return None


def code_hash(root: str) -> str:
    """Hash of the worker code under root (the image's /app, or the repo's
    worker/): every .py file under tasks/ and scripts/ plus requirements*.txt,
    by relative path and bytes."""
    digest = hashlib.sha1()
    files = []
    for sub in ("tasks", "scripts"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            files += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    files += [os.path.join(root, f) for f in ("requirements.txt", "requirements-separation.txt")
              if os.path.exists(os.path.join(root, f))]
    for path in sorted(files, key=lambda p: os.path.relpath(p, root).replace(os.sep, "/")):
        digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode() + b"\0")
        digest.update(open(path, "rb").read() + b"\0")
    return digest.hexdigest()[:12]


def record() -> dict:
    info = {
        "commit": resolve_head(BUILD_GIT_DIR) or "unknown",
        "code_hash": code_hash(IMAGE_CODE),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(BUILD_INFO, "w") as f:
        json.dump(info, f)
    return info


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-stale", action="store_true",
                        help="run even if the image wasn't built from the current checkout")


def guard(allow_stale: bool = False) -> dict:
    """Prints which code this image and the checkout hold; exits unless they
    match or allow_stale. Returns the details (for saved results)."""
    try:
        built = json.load(open(BUILD_INFO))
    except OSError:
        built = {"commit": "unknown", "code_hash": "unknown", "built_at": "unknown"}
    checkout = {"commit": resolve_head(os.path.join(REPO, ".git")),
                "code_hash": code_hash(os.path.join(REPO, "worker")) if os.path.isdir(os.path.join(REPO, "worker"))
                else None}
    problems = []
    if checkout["commit"] is None:
        problems.append(f"no checkout mounted at {REPO} to compare with (add -v <repo>:{REPO}:ro)")
    else:
        if built["commit"] != checkout["commit"]:
            problems.append(f"image built from commit {built['commit'][:10]}, checkout is at {checkout['commit'][:10]}")
        if built["code_hash"] != checkout["code_hash"]:
            problems.append("worker code in the image differs from the checkout's worker/ "
                            "(changed or uncommitted files)")
    status = "MATCHES checkout" if not problems else ("STALE (--allow-stale)" if allow_stale else "STALE")
    print(f"worker image: commit {built['commit'][:10]}, code {built['code_hash']}, built {built['built_at']}"
          f" | checkout: commit {(checkout['commit'] or 'n/a')[:10]}, code {checkout['code_hash'] or 'n/a'}"
          f" | {status}", flush=True)
    if problems and not allow_stale:
        fix = "mount the checkout" if checkout["commit"] is None else \
            "rebuild with `docker compose up -d --build worker worker-separation`"
        sys.exit("refusing to run: " + "; ".join(problems) + f". To fix, {fix}, or pass --allow-stale.")
    for p in problems:
        print(f"WARNING: {p}", flush=True)
    return {"image": built, "checkout": checkout, "stale": bool(problems)}


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        print(record())
    else:
        sys.exit("usage: python -m scripts.build_info record")
