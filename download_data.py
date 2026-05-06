"""Download the CS 3780/5780 birds Kaggle competition files.

AI/tools disclosure: This project was started with assistance from OpenAI Codex,
which helped create the Kaggle download wrapper and baseline training/submission
code. The modeling approach, experiments, final submitted code, and final Kaggle
submission should be reviewed and understood by the student before submission.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
from kagglehub.exceptions import UnauthenticatedError


COMPETITION = "cs-3780-5780-spring-2026-all-about-birds"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="data/birds", help="Where to place a local copy or symlink.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into --dest instead of creating a symlink to Kaggle's cache.",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    try:
        cache_path = Path(kagglehub.competition_download(COMPETITION))
    except UnauthenticatedError as exc:
        raise SystemExit(
            "Kaggle rejected the request because this machine is not authenticated.\n"
            "Create a Kaggle API token from Account > Settings > API, then place it at:\n"
            "  ~/.kaggle/kaggle.json\n"
            "After that, rerun: python3 download_data.py"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        print(f"{dest} already exists; leaving it unchanged.")
    elif args.copy:
        shutil.copytree(cache_path, dest)
    else:
        dest.symlink_to(cache_path, target_is_directory=True)

    print("Path to competition files:", cache_path)
    print("Project data path:", dest)


if __name__ == "__main__":
    main()

