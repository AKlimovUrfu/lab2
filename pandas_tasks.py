from __future__ import annotations

import pandas as pd
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    # Предполагаем, что data — это путь к файлу (например, 'titanic.csv') или DataFrame.
    # Если data — это строка или путь, загружаем через read_csv, иначе работаем как с DataFrame:
    df = pd.read_csv(data) if isinstance(data, (str, bytes)) else data

    # 1. Количество пропусков (всего пропущенных значений по всему датасету)
    missing_count = int(df.isnull().sum().sum())

    # 2. Число пассажиров старше 30 лет
    older_than_30 = int((df['Age'] > 30).sum())

    # 3. Средний возраст пассажиров
    average_age = float(df['Age'].mean())

    # 4. Доля выживших по классам (обычно колонка 'Pclass' и 'Survived')
    # Возвращаем в виде словаря или Series, в зависимости от контракта grader_contracts
    survival_by_class = df.groupby('Pclass')['Survived'].mean().to_dict()

    # 5. Пять наибольших тарифов по убыванию (колонка 'Fare')
    top_5_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        missing_count=missing_count,
        older_than_30=older_than_30,
        average_age=average_age,
        survival_by_class=survival_by_class,
        top_5_fares=top_5_fares,
    )

