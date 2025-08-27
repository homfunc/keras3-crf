#!/usr/bin/env python3
"""
Prepare MultiCoNER EN-English data for examples.

Usage:
  # Extract a provided zip to a target directory
  python examples/data/multiconer_prepare.py --zip examples/data/EN-English.zip --output-dir examples/data/multiconer/EN-English

  # Or copy from an existing directory
  python examples/data/multiconer_prepare.py --source-dir ../multiconer2023/EN-English --output-dir examples/data/multiconer/EN-English

This script expects the extracted directory to contain en_train.conll, en_dev.conll, en_test.conll.
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=str, help='Path to EN-English.zip (or similar) containing en_train/dev/test.conll')
    ap.add_argument('--source-dir', type=str, help='Source EN-English directory to copy from (with en_*.conll files)')
    ap.add_argument('--output-dir', required=True, help='Destination directory to write EN-English files')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.zip:
        z = Path(args.zip)
        if not z.exists():
            raise SystemExit(f'Zip not found: {z}')
        with zipfile.ZipFile(z, 'r') as zz:
            zz.extractall(out)
        print(f'Extracted {z} to {out}')
        return

    if args.source_dir:
        src = Path(args.source_dir)
        if not src.exists():
            raise SystemExit(f'Source directory not found: {src}')
        for name in ['en_train.conll', 'en_dev.conll', 'en_test.conll']:
            s = src / name
            if not s.exists():
                raise SystemExit(f'Missing expected file in source: {s}')
            shutil.copy(s, out / name)
        print(f'Copied MultiCoNER EN files to {out}')
        return

    raise SystemExit('Provide either --zip or --source-dir')


if __name__ == '__main__':
    main()

