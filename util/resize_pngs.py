#!/usr/bin/env python3
"""Batch-resize all PNGs in a dataset directory tree to 256x256.

Preserves originals by renaming the input directory with a _backup suffix,
then writes resized images to a new directory with the original name.

Usage:
    python scripts/resize_pngs.py /path/to/image_directory
"""

import argparse
import os
import sys

from PIL import Image

REQUIRED_SUBDIRS = [
    os.path.join("test", "A"),
    os.path.join("test", "B"),
    os.path.join("train", "A"),
    os.path.join("train", "B"),
    os.path.join("val", "A"),
    os.path.join("val", "B"),
]

TARGET_SIZE = (256, 256)


def main():
    parser = argparse.ArgumentParser(
        description="Resize all PNGs in a dataset directory tree to 256x256."
    )
    parser.add_argument("directory", help="Path to the dataset directory")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.directory)

    # Validate input directory exists
    if not os.path.isdir(src_dir):
        print(f"Error: '{src_dir}' is not a directory or does not exist.", file=sys.stderr)
        sys.exit(1)

    # Validate required structure
    missing = [s for s in REQUIRED_SUBDIRS if not os.path.isdir(os.path.join(src_dir, s))]
    if missing:
        print("Error: directory is missing required subdirectories:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "\nExpected structure:\n"
            f"  {os.path.basename(src_dir)}/\n"
            "  ├── test/\n"
            "  │   ├── A/\n"
            "  │   └── B/\n"
            "  ├── train/\n"
            "  │   ├── A/\n"
            "  │   └── B/\n"
            "  └── val/\n"
            "      ├── A/\n"
            "      └── B/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Backup
    backup_dir = src_dir + "_backup"
    if os.path.exists(backup_dir):
        print(f"Error: backup directory '{backup_dir}' already exists.", file=sys.stderr)
        sys.exit(1)

    print(f"Backing up: {src_dir} -> {backup_dir}")
    os.rename(src_dir, backup_dir)

    # Recreate directory structure
    for subdir in REQUIRED_SUBDIRS:
        os.makedirs(os.path.join(src_dir, subdir), exist_ok=True)

    # Walk and resize
    resized_count = 0
    skipped_count = 0

    for dirpath, _, filenames in os.walk(backup_dir):
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(src_file, backup_dir)
            dst_file = os.path.join(src_dir, rel_path)

            if not filename.lower().endswith(".png"):
                skipped_count += 1
                continue

            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            with Image.open(src_file) as img:
                resized = img.resize(TARGET_SIZE, Image.LANCZOS)
                resized.save(dst_file)

            resized_count += 1

    print(f"Done. Resized {resized_count} PNG(s) to {TARGET_SIZE[0]}x{TARGET_SIZE[1]}.")
    if skipped_count:
        print(f"Skipped {skipped_count} non-PNG file(s) (kept only in backup).")


if __name__ == "__main__":
    main()
