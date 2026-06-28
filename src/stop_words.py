"""
stop_words.py
─────────────
Loads Danish stop words from a plain-text file (one word per line).

Defaults to final_stopwords.txt in the same directory as this file.
Pass a custom path to override.
"""

from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "final_stopwords.txt"


def get_stop_words(filepath: str | Path | None = None) -> list[str]:
    """
    Load stop words from a plain-text file.

    Parameters
    ----------
    filepath : path to a .txt file with one stop word per line.
               Defaults to final_stopwords.txt next to this file.

    Returns
    -------
    List of stripped, non-empty stop word strings.
    """
    path = Path(filepath) if filepath is not None else DEFAULT_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Stop words file not found: {path}\n"
            "Either place final_stopwords.txt next to stop_words.py, "
            "or pass an explicit filepath."
        )

    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
