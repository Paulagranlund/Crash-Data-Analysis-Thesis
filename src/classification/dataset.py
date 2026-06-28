"""Tokenisation and dataset preparation for crash-narrative classification.

Built on the HuggingFace ``datasets`` library so the model inputs are produced
with a plain tokenise function and ``.map``, without a custom Dataset subclass.
"""
from datasets import Dataset


def tokenize_function(batch, tokenizer, text_col, max_length):
    """Tokenise a batch of narratives to fixed-length input_ids/attention_mask.

    Parameters
    ----------
    batch : dict
        A batch of rows provided by ``datasets.Dataset.map``.
    tokenizer :
        A HuggingFace tokenizer.
    text_col : str
        Column holding the narrative text.
    max_length : int
        Truncation and padding length.
    """
    return tokenizer(
        batch[text_col],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def build_dataset(df, tokenizer, text_col, label_col, max_length):
    """Build a tokenised HuggingFace Dataset from a dataframe.

    The label column is exposed as ``labels`` so the Trainer can read it, the
    raw text column is dropped after tokenisation, and the result is formatted
    as torch tensors.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame holding the text and the integer-encoded label.
    tokenizer :
        A HuggingFace tokenizer.
    text_col : str
        Column holding the narrative text.
    label_col : str
        Column holding the integer-encoded label.
    max_length : int
        Truncation and padding length.

    Returns
    -------
    datasets.Dataset
        Tokenised dataset with input_ids, attention_mask, token_type_ids
        (where produced) and labels.
    """
    ds = Dataset.from_pandas(
        df[[text_col, label_col]].rename(columns={label_col: "labels"}),
        preserve_index=False,
    )
    ds = ds.map(
        lambda batch: tokenize_function(batch, tokenizer, text_col, max_length),
        batched=True,
    )
    ds = ds.remove_columns([text_col])
    ds.set_format(type="torch")
    return ds
