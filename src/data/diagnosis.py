"""
src/data/diagnosis.py

Utilities for grouping ICD-9 diagnosis codes into clinically meaningful
disease categories.

The original dataset contains hundreds of unique ICD-9 diagnosis codes.
Grouping them into broader categories reduces cardinality while preserving
clinical relevance for machine learning models.
"""

from __future__ import annotations

import pandas as pd


UNKNOWN = "Unknown"
OTHER = "Other"


def group_diagnosis_codes(code: object) -> str:
    """
    Group an ICD-9 diagnosis code into a broader disease category.

    Parameters
    ----------
    code : object
        Raw ICD-9 diagnosis code.

    Returns
    -------
    str
        Disease category.
    """

    if pd.isna(code):
        return UNKNOWN

    code = str(code).strip()

    if code.startswith(("V", "E")):
        return OTHER

    try:
        code = int(float(code.split(".")[0]))
    except (TypeError, ValueError):
        return OTHER

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"

    if 460 <= code <= 519 or code == 786:
        return "Respiratory"

    if 520 <= code <= 579 or code == 787:
        return "Digestive"

    if code == 250:
        return "Diabetes"

    if 800 <= code <= 999:
        return "Injury"

    if 710 <= code <= 739:
        return "Musculoskeletal"

    if 580 <= code <= 629 or code == 788:
        return "Genitourinary"

    if 140 <= code <= 239:
        return "Neoplasms"

    return OTHER