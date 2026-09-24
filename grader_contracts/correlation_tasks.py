from dataclasses import dataclass

@dataclass(frozen=True)
class BrainDataInput:
    csv_path: str

@dataclass(frozen=True)
class BrainCorrelationSummary:
    men_count: int
    women_count: int
    women_mri_correlation: dict[str, float]
    men_mri_correlation: dict[str, float]
    strongest_mri_feature: str
