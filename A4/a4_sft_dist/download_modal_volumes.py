"""Download model checkpoints and eval results from Modal volumes to local disk.

Requires the `modal` package and a valid Modal authentication token
(run `modal setup` or `scripts/setup_modal.sh` first).

Usage examples:
    # Download everything (checkpoints + eval results)
    python3 download_modal_volumes.py

    # Download only eval results
    python3 download_modal_volumes.py --volume eval

    # Download a specific training run
    python3 download_modal_volumes.py --volume output --remote-prefix gsm_symbolic_train_4500_student-gemma-3-1b-it

    # List remote files without downloading
    python3 download_modal_volumes.py --dry-run
"""

import argparse
import os
from pathlib import Path

import modal
from modal.volume import FileEntryType

# (volume_name, local_subdir)
VOLUME_MAP = {
    "output": ("a4-output", "out"),
    "eval": ("a4-eval-results", "eval_results"),
}


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download_volume(
    volume_name: str,
    local_subdir: str,
    local_dir: str,
    remote_prefix: str | None,
    dry_run: bool,
) -> None:
    vol = modal.Volume.from_name(volume_name)

    list_path = f"/{remote_prefix}" if remote_prefix else "/"
    print(f"\n[{volume_name}] Listing files under '{list_path}' ...")

    entries = vol.listdir(list_path, recursive=True)

    num_downloaded = 0
    num_skipped = 0
    total_bytes = 0

    for entry in entries:
        remote_path = entry.path
        if entry.type != FileEntryType.FILE:
            continue

        local_path = Path(local_dir) / local_subdir / remote_path.lstrip("/")

        size = getattr(entry, "size", None)
        size_str = _human_size(size) if size is not None else "?"

        # Skip if local file already matches remote size
        if not dry_run and local_path.exists() and size is not None:
            if local_path.stat().st_size == size:
                num_skipped += 1
                continue

        if dry_run:
            print(f"  [DRY-RUN] {remote_path}  ({size_str})")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(remote_path):
                f.write(chunk)

        file_bytes = local_path.stat().st_size
        total_bytes += file_bytes
        num_downloaded += 1
        print(f"  Downloaded: {remote_path}  ({_human_size(file_bytes)})")

    if dry_run:
        print(f"[{volume_name}] Dry-run complete. {len(entries)} entries listed.")
    else:
        print(
            f"[{volume_name}] Done. "
            f"{num_downloaded} downloaded ({_human_size(total_bytes)}), "
            f"{num_skipped} skipped (already exist)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model checkpoints and eval results from Modal volumes."
    )
    parser.add_argument(
        "--volume",
        choices=["output", "eval", "all"],
        default="all",
        help="Which volume(s) to download (default: all).",
    )
    parser.add_argument(
        "--remote-prefix",
        default=None,
        help="Only download files under this subdirectory within the volume.",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="Local destination root (default: current directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List remote files without downloading.",
    )
    args = parser.parse_args()

    if args.volume == "all":
        volumes = list(VOLUME_MAP.items())
    else:
        volumes = [(args.volume, VOLUME_MAP[args.volume])]

    for _key, (volume_name, local_subdir) in volumes:
        download_volume(
            volume_name=volume_name,
            local_subdir=local_subdir,
            local_dir=args.local_dir,
            remote_prefix=args.remote_prefix,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
