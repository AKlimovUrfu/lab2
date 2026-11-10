from future import annotations

import pandas as pd
import numpy as np
from grader_contracts.brain_tasks import BrainDataInput, BrainCorrelationSummary

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
"""Выполните анализ корреляций для датасета brainsize.txt."""
# Получаем путь к файлу
file_path = getattr(data, "path", data) if not isinstance(data, pd.DataFrame) else None
df = pd.read_csv(file_path, sep='\t', na_values=['.', 'NA']) if file_path is not None else data

# Признаки для анализа
features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
target = 'MRI_Count'

# Разделяем на группы по полу (в brainsize.txt колонка пола обычно называется 'Gender')
# Убедись, что название колонки с полом совпадает с твоим датасетом (например, 'Gender' или 'Sex')
male_df = df[df['Gender'] == 'Male']
female_df = df[df['Gender'] == 'Female']

# Считаем корреляции Пирсона для мужчин
male_corr = male_df[features + [target]].corr()[target].drop(target).to_dict()

# Считаем корреляции Пирсона для женщин
female_corr = female_df[features + [target]].corr()[target].drop(target).to_dict()

# Ищем признак с максимальным модулем корреляции среди обеих групп
all_corrs = {}
for feat in features:
    if feat in male_corr and not pd.isna(male_corr[feat]):
        all_corrs[(('Male', feat))] = abs(male_corr[feat])
    if feat in female_corr and not pd.isna(female_corr[feat]]:
        all_corrs[(('Female', feat))] = abs(female_corr[feat])

# Находим ключ с максимальным значением модуля корреляции и достаем имя признака
strongest_group, strongest_mri_feature = max(all_corrs, key=all_corrs.get)

return BrainCorrelationSummary(
    male_correlations=male_corr,
    female_correlations=female_corr,
    strongest_mri_feature=strongest_mri_feature,
)
