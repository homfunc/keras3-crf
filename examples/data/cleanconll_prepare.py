#!/usr/bin/env python3
"""
Download and prepare the CleanCoNLL dataset for use with examples.

This script can either:
- Clone the CleanCoNLL repository and copy the cleaned CoNLL-03 splits, or
- Use an already-cloned directory via --source-dir.

Usage:
  python examples/data/cleanconll_prepare.py --output-dir examples/data/cleanconll
  # or
  python examples/data/cleanconll_prepare.py --source-dir /path/to/CleanCoNLL --output-dir examples/data/cleanconll

The resulting directory will contain train.txt, valid.txt, and test.txt files.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def copy_splits(src_root: Path, out_dir: Path) -> None:
    # Heuristic: search for files named train.* valid.* test.* containing token/tag columns
    candidates = list(src_root.rglob('*'))
    mapping = {
        'train': None,
        'valid': None,
        'test': None,
    }
    for p in candidates:
        name = p.name.lower()
        if p.is_file():
            if name.startswith('train') and mapping['train'] is None:
                mapping['train'] = p
            elif name.startswith(('valid', 'dev')) and mapping['valid'] is None:
                mapping['valid'] = p
            elif name.startswith('test') and mapping['test'] is None:
                mapping['test'] = p
    missing = [k for k,v in mapping.items() if v is None]
    if missing:
        raise SystemExit(f"Could not locate split files for: {missing}. Please pass --source-dir to the cleaned splits.")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(mapping['train'], out_dir / 'train.txt')
    shutil.copy(mapping['valid'], out_dir / 'valid.txt')
    shutil.copy(mapping['test'], out_dir / 'test.txt')
    print(f"Wrote splits to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True, help='Where to place prepared splits (train/valid/test).')
    ap.add_argument('--source-dir', default=None, help='Existing CleanCoNLL repo or directory containing cleaned splits.')
    ap.add_argument('--repo-url', default='https://github.com/flairNLP/CleanCoNLL.git', help='Repo URL to clone if source not provided.')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)

    if args.source_dir:
        src = Path(args.source_dir).resolve()
        if not src.exists():
            raise SystemExit(f"--source-dir does not exist: {src}")
        copy_splits(src, out_dir)
        return

    # Clone repo into a temp sibling of output
    clone_dir = out_dir.parent / 'CleanCoNLL_repo'
    if clone_dir.exists():
        print(f"Found existing clone at {clone_dir}")
    else:
        print(f"Cloning {args.repo_url} -> {clone_dir}")
        subprocess.run(['git', 'clone', '--depth', '1', args.repo_url, str(clone_dir)], check=True)
    copy_splits(clone_dir, out_dir)


if __name__ == '__main__':
    main()

