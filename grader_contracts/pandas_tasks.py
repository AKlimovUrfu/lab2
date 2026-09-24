from dataclasses import dataclass

@dataclass(frozen=True)
class TitanicInput:
    csv_path: str

@dataclass(frozen=True)
class TitanicSummary:
    row_count: int
    missing_by_column: dict[str, int]
    adults_over_30_count: int
    mean_age_by_pclass: dict[int, float]
    survival_rate_by_pclass: dict[int, float]
    highest_fares: list[float]
