"""
src/data/split.py

Utilities for performing patient-level train/test splits.

The Diabetes 130-US Hospitals dataset contains multiple encounters
for the same patient. Splitting by rows would introduce data leakage,
since encounters from the same patient could appear in both the
training and testing sets.

This module performs splitting at the patient level.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split


def patient_level_split(
    df: pd.DataFrame,
    patient_column: str = "patient_nbr",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split a dataframe by unique patients rather than encounters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    patient_column : str, default="patient_nbr"
        Column identifying each patient.

    test_size : float, default=0.2
        Fraction of patients allocated to the test set.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame

    test_df : pd.DataFrame
    """

    unique_patients = df[patient_column].unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = (
        df[df[patient_column].isin(train_patients)]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[df[patient_column].isin(test_patients)]
        .copy()
        .reset_index(drop=True)
    )

    return train_df, test_df


def leakage_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_column: str = "patient_nbr",
) -> bool:
    """
    Check whether any patient appears in both train and test sets.

    Returns
    -------
    bool
        True if leakage exists.
        False otherwise.
    """

    train_patients = set(train_df[patient_column])

    test_patients = set(test_df[patient_column])

    overlap = train_patients.intersection(test_patients)

    return len(overlap) > 0


def split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_column: str = "patient_nbr",
):
    """
    Print a summary of the train/test split.
    """

    print("=" * 60)
    print("Patient-Level Train/Test Split Summary")
    print("=" * 60)

    print(f"Train Encounters : {len(train_df):,}")
    print(f"Test Encounters  : {len(test_df):,}")

    print()

    print(f"Train Patients   : {train_df[patient_column].nunique():,}")
    print(f"Test Patients    : {test_df[patient_column].nunique():,}")

    print()

    leakage = leakage_check(
        train_df,
        test_df,
        patient_column,
    )

    print(f"Data Leakage Detected : {leakage}")