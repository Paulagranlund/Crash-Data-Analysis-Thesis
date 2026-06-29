import os
from typing import Dict, Optional, Tuple, Any
import pandas as pd

COLUMN_RENAME_MAP = {
    "UHELDSDATO": "accident_date",
    "UHELDSART": "report_category",
    "KODE_UHELDSSITUATION": "encoded_accident_situation",
    "UHELDSSITUATION": "accident_situation",
    "UHELDSTEKST": "police_narrative",
    "AAR": "year",
    "UHELDS_ID": "accident_id",
}

def load_and_clean_accident_data(
    data_folder: str,
    subset_size: Optional[int] = None,
    encode_labels: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[int, int]], Optional[Dict[int, int]]]:

    """
    Load and clean accident data from Excel files.

    Parameters:
    data_folder : str
        Path to the folder containing the Excel files.
    subset_size : Optional[int], default=None
        If given, randomly sample this many rows from the cleaned dataset.
        If None, return the full cleaned dataset.
    encode_labels : bool, default=False
        If True, encode main_situation_class to consecutive integers, which is needed for BERT classification
    verbose : bool, default=True
        If True, print progress information.
    
    Returns:
    df : pd.DataFrame
        The cleaned dataframe.
    label2id : dict or None
        Mapping from original labels to encoded labels, if encode_labels=True.
    id2label : dict or None
        Reverse mapping from encoded labels to original labels, if encode_labels=True.
    """

    if verbose:
        print("Loading all Excel files...")

    all_dfs = []
    required_columns = set(COLUMN_RENAME_MAP)

    excel_paths = []
    for dirpath, _, filenames in os.walk(data_folder):
        for file in filenames:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                excel_paths.append(os.path.join(dirpath, file))

    for path in sorted(excel_paths):
        rel_path = os.path.relpath(path, data_folder)
        if verbose:
            print(f"Loading: {rel_path}")
        df_temp = pd.read_excel(path, header=2)
        missing = required_columns.difference(df_temp.columns)
        if missing:
            if verbose:
                print(f"  skipping {rel_path}: missing accident columns {sorted(missing)}")
            continue
        df_temp = df_temp.rename(columns=COLUMN_RENAME_MAP)
        df_temp = df_temp[list(COLUMN_RENAME_MAP.values())]
        all_dfs.append(df_temp)

    if not all_dfs:
        raise FileNotFoundError(
            f"No accident-level .xlsx files with columns {sorted(required_columns)} "
            f"found under folder: {data_folder}"
        )

    df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"Combined dataset shape: {df.shape}")

    # Keep all years up to and including 2025
    df = df[df["year"] <= 2025].copy()

    # Convert encoded accident situation to numeric
    df["encoded_accident_situation"] = pd.to_numeric(
        df["encoded_accident_situation"], errors="coerce"
    )

    # Main class label: e.g. 201 -> 2
    df["main_situation_class"] = (
        df["encoded_accident_situation"] // 100
    ).astype("Int64")

    # Remove missing labels and narratives
    df = df[df["main_situation_class"].notna()].copy()
    df = df[df["police_narrative"].notna()].copy()

    # Remove very short narratives
    df["n_words"] = df["police_narrative"].str.split().str.len()
    df = df[df["n_words"] >= 3].copy()

    if verbose:
        print(f"After cleaning: {df.shape}")

    # Optional subset for quick testing
    if subset_size is not None:
        if subset_size > len(df):
            if verbose:
                print(
                    f"Requested subset_size={subset_size}, but cleaned dataset only has {len(df)} rows. Using full dataset."
                )
        else:
            df = df.sample(n=subset_size, random_state=42).copy()
            if verbose:
                print(f"Subset selected: {df.shape}")

    label2id = None
    id2label = None

    # Optional label encoding for classification models
    if encode_labels:
        unique_labels = sorted(int(x) for x in df["main_situation_class"].unique())
        label2id = {orig: i for i, orig in enumerate(unique_labels)}
        id2label = {i: orig for orig, i in label2id.items()}
        df["main_situation_class"] = (
            df["main_situation_class"].map(label2id).astype(int)
        )

        if verbose:
            print("Label mapping:", label2id)

    return df, label2id, id2label

def load_data_for_modeling(
    data_folder: str,
    subset_size: Optional[int] = None,
    encode_labels: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:

    """
    Convenience wrapper for loading data and preparing docs/labels.

    Parameters:
    data_folder : str
        Path to the folder containing the Excel files.
    subset_size : Optional[int], default=None
        If given, randomly sample this many rows from the cleaned dataset.
    encode_labels : bool, default=False
        If True, encode main_situation_class to consecutive integers.
    verbose : bool, default=True
        If True, print progress information.

    Returns:
    dict
        Dictionary containing:
        - df
        - docs
        - labels
        - label2id
        - id2label
    """

    df, label2id, id2label = load_and_clean_accident_data(
        data_folder=data_folder,
        subset_size=subset_size,
        encode_labels=encode_labels,
        verbose=verbose,
    )

    docs = df["police_narrative"].tolist()
    labels = df["main_situation_class"].tolist()

    if verbose:
        print(f"\nTotal documents for modeling: {len(docs)}")
        print(f"Unique labels: {sorted(set(labels))}")

    return {
        "df": df,
        "docs": docs,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
    }



"""
How to load for BERTopic:

from data_load import load_data_for_modeling

data = load_data_for_modeling(
    data_folder="/mnt/raid/data_safetyLLMthesis/data",
    subset_size=10000,   # optional
    encode_labels=False,
    verbose=True,
)

df = data["df"]
docs = data["docs"]
labels = data["labels"]


How to use for BERT classification:

from data_load import load_data_for_modeling

data = load_data_for_modeling(
    data_folder="/mnt/raid/data_safetyLLMthesis/data",
    encode_labels=True,
    verbose=True,
)

df = data["df"]
docs = data["docs"]
labels = data["labels"]
label2id = data["label2id"]
id2label = data["id2label"]
"""