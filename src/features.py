"""Feature engineering: ICD-9 chapter grouping, target binarization,
and other clinical transformations.
"""
from __future__ import annotations

import pandas as pd


def binarize_readmission(series: pd.Series) -> pd.Series:
    """Convert the 3-class target into binary.

    1 = readmission within 30 days (clinical outcome of interest)
    0 = readmission after 30 days OR no readmission
    """
    return (series == "<30").astype(int)


# ICD-9-CM chapter boundaries (numeric prefix → chapter name).
# Source: official ICD-9-CM tabular list.
# Special codes (V-codes for health-status factors, E-codes for external causes)
# get their own categories.
ICD9_CHAPTERS = [
    (1, 139, "Infectious"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine"),         # includes 250.x diabetes
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms"),          # ill-defined conditions
    (800, 999, "Injury"),
]


def group_icd9(code: object) -> str:
    """Map a single ICD-9 code (as stored in this dataset) to a clinical chapter.

    The dataset stores codes as strings like "250.83", "428.0", "V58", "E885".
    V-codes (factors influencing health status) and E-codes (external causes)
    are grouped into their own categories.
    """
    if pd.isna(code):
        return "Missing"

    code_str = str(code).strip()

    # V-codes: factors influencing health status (e.g., V58 = aftercare)
    if code_str.startswith("V"):
        return "V_code"

    # E-codes: external causes of injury (e.g., E885 = fall)
    if code_str.startswith("E"):
        return "E_code"

    # Numeric codes: take the integer part before the decimal
    try:
        numeric = int(float(code_str))
    except (ValueError, TypeError):
        return "Unknown"

    for low, high, chapter in ICD9_CHAPTERS:
        if low <= numeric <= high:
            return chapter

    return "Unknown"