from future import annotations

import pandas as pd
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary

def analyze_titanic(data: TitanicInput) -> TitanicSummary:
"""Выполните загрузку и анализ датасета Titanic."""
# 1. Загрузить CSV в DataFrame
# Если TitanicInput передает путь через атрибут или является строкой/путь:
file_path = getattr(data, "path", data) if not isinstance(data, pd.DataFrame) else None
df = pd.read_csv(file_path) if file_path is not None else data

# 2. Посчитать количество пропусков в каждом столбце (в виде словаря или Series)
missing_count = df.isnull().sum().to_dict()

# 3. Число пассажиров старше 30 лет
older_than_30 = int((df['Age'] > 30).sum())

# 4. Вычислить средний возраст для каждого значения Pclass
average_age = df.groupby('Pclass')['Age'].mean().to_dict()

# 5. Вычислить долю выживших для каждого Pclass (ключи — целые числа)
survival_by_class = {int(k): v for k, v in df.groupby('Pclass')['Survived'].mean().to_dict().items()}

# 6. Вернуть пять наибольших значений Fare в порядке убывания (tolist() или список)
top_5_fares = df['Fare'].nlargest(5).tolist()

return TitanicSummary(
    missing_count=missing_count,
    older_than_30=older_than_30,
    average_age=average_age,
    survival_by_class=survival_by_class,
    top_5_fares=top_5_fares,
)
