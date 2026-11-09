"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt."""
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    sex_col = "Sex"
    if df[sex_col].dtype in ("int64", "float64"):
        male_df = df[df[sex_col] == 1]
        female_df = df[df[sex_col] == 2]
    else:
        male_df = df[df[sex_col].str.strip().str.lower() == "male"]
        female_df = df[df[sex_col].str.strip().str.lower() == "female"]

    men_count = int(len(male_df))
    women_count = int(len(female_df))

    men_mri_correlation = {
        feat: float(male_df[feat].corr(male_df["MRI_Count"]))
        for feat in features
    }
    women_mri_correlation = {
        feat: float(female_df[feat].corr(female_df["MRI_Count"]))
        for feat in features
    }

    strongest_mri_feature = max(
        features,
        key=lambda f: max(
            abs(men_mri_correlation[f]),
            abs(women_mri_correlation[f]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )