"""Create a .tar.gz backup copy of the stored ArXiv data.

Backs up data/rss/ (raw snapshots — never delete these) and data/papers/
(parsed JSON) into a single compressed archive under backups/, which is
gitignored: data stays out of the git repo but is preserved on disk.

Usage:
    uv run python backup_data.py [data_dir] [out_dir]
    # defaults: data -> backups
"""

from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

# subdirectories of data_dir to include in the backup
SUBDIRS = ("rss", "papers")


def backup(data_dir: str | Path = "data", out_dir: str | Path = "backups") -> Path:
    """Pack data/rss/ and data/papers/ into backups/<stamp>.tar.gz.

    Returns the path of the created archive. Skips subdirectories that do
    not exist; fails if none exist.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M")
    path = out_dir / f"arxiv-data-{stamp}.tar.gz"

    included = 0
    raw_bytes = 0
    with tarfile.open(path, "w:gz") as tar:
        for sub in SUBDIRS:
            src = data_dir / sub
            if not src.is_dir():
                continue
            for file in sorted(src.iterdir()):
                if file.is_file():
                    tar.add(file, arcname=f"{sub}/{file.name}")
                    included += 1
                    raw_bytes += file.stat().st_size

    if included == 0:
        path.unlink()
        raise SystemExit(f"nothing to back up under {data_dir} ({', '.join(SUBDIRS)})")

    print(f"archive : {path}")
    print(f"files   : {included} ({', '.join(SUBDIRS)})")
    print(f"raw     : {raw_bytes / 1e6:.1f} MB -> tar.gz {path.stat().st_size / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    backup(
        sys.argv[1] if len(sys.argv) > 1 else "data",
        sys.argv[2] if len(sys.argv) > 2 else "backups",
    )
