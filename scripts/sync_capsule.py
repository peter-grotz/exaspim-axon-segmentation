"""Render the Code Ocean capsule from ``capsule/`` and vendor the package at the current commit.

Usage::

    uv run python scripts/sync_capsule.py <path to a git clone of the capsule>

The capsule's working tree is replaced except for ``KEEP``: git metadata, Code Ocean secrets and
data-asset attachments (managed in the Code Ocean UI), and notebooks in ``code/``. Review
``git -C <capsule> diff`` and commit/push there. Requires a clean, pushed package working tree so
``SOURCE_COMMIT`` names a commit that exists on GitHub.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

from exaspim_axon_segmentation.io.ontology import STRUCTURE_GRAPH_URL
from exaspim_axon_segmentation.model import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "capsule"
PACKAGE_FILES = ["pyproject.toml", "uv.lock", "README.md", "LICENSE", "src"]
KEEP = (
    ".git",
    ".codeocean/secrets.json",
    ".codeocean/datasets.json",
    "environment/environment.yml",  # Code Ocean credentials (newer format of secrets.json)
    ".codeocean/environment.json",  # written by Code Ocean after each environment build
    "metadata/metadata.yml",  # capsule name/description edited in the Code Ocean UI
    "code/*.ipynb",
)
WEIGHTS = "trailmap-axon-210409"


def git(*args: str) -> str:
    """Run git in the package repo and return stdout."""
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_and_pushed() -> str:
    """Return HEAD if the tree is clean and HEAD is on the upstream branch."""
    if git("status", "--porcelain"):
        sys.exit("Package working tree is dirty; commit first.")
    head = git("rev-parse", "HEAD")
    git("fetch", "--quiet")
    if not git("branch", "-r", "--contains", head):
        sys.exit(f"{head[:8]} is not pushed; push first.")
    return head


def _kept(rel: str) -> bool:
    return any(fnmatch(rel, k) or rel.startswith(f"{k}/") for k in KEEP)


def clear(capsule: Path) -> None:
    """Delete everything in the capsule except the kept paths."""
    for path in capsule.rglob("*"):
        if path.is_file() and not _kept(path.relative_to(capsule).as_posix()):
            path.unlink()
    directories = sorted((p for p in capsule.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts))
    for path in directories:
        if not _kept(path.relative_to(capsule).as_posix()) and not any(path.iterdir()):
            path.rmdir()


def render_post_install() -> str:
    """Fill the postInstall template with pinned requirements and weight coordinates.

    Deliberately independent of the package commit, so the image only rebuilds when
    dependencies, weights, or the template change.
    """
    requirements = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    pins = "\n".join(line for line in requirements.splitlines() if line and line[0] not in "# ")
    weights = REGISTRY[WEIGHTS]
    replacements = {
        "@REQUIREMENTS@": pins,
        "@HF_REPO@": weights.repo_id,
        "@HF_FILE@": weights.filename,
        "@HF_REVISION@": weights.revision,
        "@ONTOLOGY_URL@": STRUCTURE_GRAPH_URL,
    }
    text = (TEMPLATE / "environment" / "postInstall.in").read_text()
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def main() -> None:
    """Sync the capsule clone given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capsule", type=Path, help="git clone of the Code Ocean capsule")
    capsule = parser.parse_args().capsule.resolve()
    if not (capsule / ".git").exists():
        sys.exit(f"{capsule} is not a git clone")
    commit = require_clean_and_pushed()

    clear(capsule)
    for src in TEMPLATE.rglob("*"):
        rel = src.relative_to(TEMPLATE)
        if src.is_dir() or rel.name == "postInstall.in":
            continue
        dst = capsule / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    post_install = capsule / "environment" / "postInstall"
    post_install.write_text(render_post_install())
    post_install.chmod(0o755)

    vendored = capsule / "code" / "exaspim-axon-segmentation"
    for name in PACKAGE_FILES:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, vendored / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            vendored.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, vendored / name)
    remote = git("remote", "get-url", "origin")
    (vendored / "SOURCE_COMMIT").write_text(f"{remote}@{commit}\n")
    print(f"Synced {capsule} to {commit[:12]}; review with: git -C {capsule} status")


if __name__ == "__main__":
    main()
