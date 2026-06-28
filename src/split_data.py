from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "main_situation_class",
    test_size: float = 0.2,
    val_size_within_trainval: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into stratified train, validation, and test sets.

    Parameters:
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Column used for stratification.
    test_size : float
        Proportion of full data used for test set.
    val_size_within_trainval : float
        Proportion of the remaining train+val set used for validation.
        Example: 0.1 means 10% of trainval, not 10% of full dataset.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print split sizes.

    Returns:
    df_train, df_val, df_test : tuple of pd.DataFrame
        Stratified train, validation, and test sets.
    """
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_within_trainval,
        random_state=random_state,
        stratify=df_trainval[label_col],
    )

    if verbose:
        print(
            f"Train samples: {len(df_train):,} | "
            f"Validation samples: {len(df_val):,} | "
            f"Test samples: {len(df_test):,}"
        )

    return df_train, df_val, df_test

"""
How to use for BERT classification:

from split_data import stratified_train_val_test_split

LABEL_COL = "main_situation_class"

df_train, df_val, df_test = stratified_train_val_test_split(
    df=df,
    label_col=LABEL_COL,
    random_state=SEED,
    verbose=True,
)
"""