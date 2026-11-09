"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = data.df

    df_male = df[df['Gender'] == 'Male']
    df_female = df[df['Gender'] == 'Female']

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height', 'MRI_Count']

    corr_male = df_male[features].corr(method='pearson')
    corr_female = df_female[features].corr(method='female')

    mri_corr_male = corr_male['MRI_Count'].drop('MRI_Count')
    mri_corr_female = corr_female['MRI_Count'].drop('MRI_Count')

    combined_mri_corr = pd.concat([mri_corr_male, mri_corr_female])
    strongest_feature = combined_mri_corr.abs().idxmax()

    return BrainCorrelationSummary(
        male_correlation=corr_male,
        female_correlation=corr_female,
        strongest_mri_feature=strongest_feature
    )
