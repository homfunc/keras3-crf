from typing import List, Tuple
from pathlib import Path


def read_multiconer_en_splits(data_dir: str, token_col: int = 0, tag_col: int = -1) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Read MultiCoNER EN-English subset with expected filenames:
      - en_train.conll
      - en_dev.conll
      - en_test.conll
    Returns: (train_s, train_t, dev_s, dev_t, test_s, test_t)
    """
    from .data import read_conll
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"MultiCoNER directory not found: {data_dir}")
    f_train = p / "en_train.conll"
    f_dev = p / "en_dev.conll"
    f_test = p / "en_test.conll"
    for f in (f_train, f_dev, f_test):
        if not f.exists():
            raise FileNotFoundError(f"Expected file not found: {f}")
    train_s, train_t = read_conll(str(f_train), token_col=token_col, tag_col=tag_col, lowercase=False)
    dev_s, dev_t = read_conll(str(f_dev), token_col=token_col, tag_col=tag_col, lowercase=False)
    test_s, test_t = read_conll(str(f_test), token_col=token_col, tag_col=tag_col, lowercase=False)
    return train_s, train_t, dev_s, dev_t, test_s, test_t

