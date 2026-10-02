"""Unit tests for data cleaning and validation rules."""

import os
import pytest
import pandas as pd
import numpy as np
from src.cleaning import validate_and_clean_data, ALLOWED_RANGES, CANONICAL_COLUMNS


def test_clean_data_exists_and_valid():
    """Verify that interim cleaned data satisfies all canonical constraints."""
    interim_path = "data/interim/students_clean.csv"
    assert os.path.exists(interim_path), "Cleaned dataset does not exist"
    
    df = pd.read_csv(interim_path)
    assert len(df) > 0, "Cleaned dataset is empty"
    
    # Check all columns exist
    for col in CANONICAL_COLUMNS:
        assert col in df.columns, f"Missing canonical column {col}"
        
    # Check zero missing values across features and target
    assert df.isna().sum().sum() == 0, "Missing values found in cleaned dataset"
    
    # Check ranges
    for col, (low, high) in ALLOWED_RANGES.items():
        assert (df[col] >= low).all(), f"Values below minimum in {col}"
        assert (df[col] <= high).all(), f"Values above maximum in {col}"
        
    # Check uniqueness of student_id
    assert df["student_id"].nunique() == len(df), "Duplicate student_ids present"
