from typing import List, Tuple, Dict
from pathlib import Path

def read_multiconer_dir(data_dir: str, token_col: int = 0, tag_col: int = -1) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read MultiCoNER-style files from a directory and return (sentences, tags).

    Heuristics:
    - Looks for files containing whitespace-separated columns with a tag column.
    - Splits sentences on blank lines.
    - Intended for files like train.txt/dev.txt/test.txt in the EN_-English subset.
    """
    from .data import read_conll
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"MultiCoNER directory not found: {data_dir}")
    # Prefer standard split filenames if present
    candidates = [
        p / 'train.txt', p / 'dev.txt', p / 'valid.txt', p / 'validation.txt', p / 'test.txt'
    ]
    files = [f for f in candidates if f.exists()]
    if not files:
        # fallback: any .txt files
        files = sorted([pp for pp in p.glob('*.txt') if pp.is_file()])
    if not files:
        # fallback: any files
        files = sorted([pp for pp in p.iterdir() if pp.is_file()])
    # Concatenate all into one big list of sentences; users can split outside if needed
    all_sentences: List[List[str]] = []
    all_tags: List[List[str]] = []
    for f in files:
        s, t = read_conll(str(f), token_col=token_col, tag_col=tag_col, lowercase=False)
        all_sentences.extend(s)
        all_tags.extend(t)
    return all_sentences, all_tags

