"""
src/data/specialty.py

Utilities for grouping medical specialties into broader clinical categories.
"""

from __future__ import annotations

import pandas as pd


MISSING = "Missing"
OTHER = "Other Specialty"


def group_medical_specialty(specialty: object) -> str:
    """
    Group medical specialties into broader clinical categories.

    Parameters
    ----------
    specialty : object
        Raw medical specialty.

    Returns
    -------
    str
        Grouped medical specialty.
    """

    if pd.isna(specialty):
        return MISSING

    specialty = str(specialty).strip().lower()

    if specialty in {"?", "unknown", "not available"}:
        return MISSING

    if (
        "surgeon" in specialty
        or "surgery" in specialty
        or specialty in {"orthopedics", "urology"}
    ):
        return "Surgery"

    if "cardio" in specialty:
        return "Cardiology"

    if specialty in {
        "internalmedicine",
        "internal medicine",
        "endocrinology",
        "gastroenterology",
        "nephrology",
        "pulmonology",
    }:
        return "Internal Medicine"

    if (
        "emergency" in specialty
        or "critical care" in specialty
        or "icu" in specialty
    ):
        return "Emergency/Critical Care"

    if (
        "family" in specialty
        or "general" in specialty
        or specialty in {"practitioner", "primary care"}
    ):
        return "General Practice"

    if (
        "pediatric" in specialty
        or specialty == "neonatology"
    ):
        return "Pediatrics"

    if (
        "obgyn" in specialty
        or "obstetrics" in specialty
        or "gynecology" in specialty
    ):
        return "OB-GYN"

    if (
        "psych" in specialty
        or "neurology" in specialty
    ):
        return "Mental/Neurological Health"

    return OTHER