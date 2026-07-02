"""
src/data/cleaning.py

General data cleaning utilities.

These functions are dataset-agnostic and can be reused across
multiple machine learning projects.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def replace_missing_values(
    df: pd.DataFrame,
    placeholder: str = "?"
) -> pd.DataFrame:
    """
    Replace placeholder values with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    placeholder : str, default="?"
        Placeholder used for missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame with placeholder values replaced by NaN.
    """
    df = df.copy()
    df.replace(placeholder, np.nan, inplace=True)
    return df


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a missing value summary.

    Returns
    -------
    pd.DataFrame
        Missing count and percentage for every column.
    """

    summary = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing Percentage": (df.isna().mean() * 100).round(2)
    })

    summary = (
        summary
        .sort_values("Missing Percentage", ascending=False)
    )

    return summary


def remove_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """
    Remove specified columns.

    Parameters
    ----------
    df : pd.DataFrame

    columns : list
        Columns to remove.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    existing_columns = [col for col in columns if col in df.columns]

    return df.drop(columns=existing_columns)

def create_binary_target(
    df: pd.DataFrame,
    source_column: str = "readmitted",
    target_column: str = "readmitted_binary"
) -> pd.DataFrame:
    """
    Convert the original readmission target
    into a binary target.

    Mapping

    NO   -> 0
    >30  -> 0
    <30  -> 1
    """

    mapping = {
        "NO": 0,
        ">30": 0,
        "<30": 1
    }

    df = df.copy()

    df[target_column] = df[source_column].map(mapping)

    return df