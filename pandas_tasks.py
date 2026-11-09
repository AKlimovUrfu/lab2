"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations
import pandas as pd
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary

def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = data.df

    missing_values = int(df.isna().sum().sum())
    passengers_over_30 = int((df['Age'] > 30).sum())
    mean_age = float(df['Age'].mean())

    survival_rates = df.groupby('Pclass')['Survived'].mean().to_dict()
    top_5_fares = df['Fare'].sort_values(ascending=False).head(5).tolist()

    return TitanicSummary(
        missing_values=missing_values,
        passengers_over_30=passengers_over_30,
        mean_age=mean_age,
        survival_rates=survival_rates,
        top_5_fares=top_5_fares
    )
