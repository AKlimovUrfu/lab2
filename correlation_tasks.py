from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def load_brain_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t", na_values=["NA"])
    df["Gender"] = df["Gender"].astype(str).str.strip()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    return {
        feature: float(group[feature].corr(group[TARGET], method="pearson"))
        for feature in FEATURES
    }


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = load_brain_data(data.csv_path)
    gender = df["Gender"].str.lower()
    men = df[gender == "male"]
    women = df[gender == "female"]

    men_corr = mri_correlations(men)
    women_corr = mri_correlations(women)

    candidates = [
        (abs(value), feature)
        for corr in (women_corr, men_corr)
        for feature, value in corr.items()
        if pd.notna(value)
    ]
    strongest_mri_feature = max(candidates)[1]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )


def plot_brain_data(csv_path: str = "brainsize.txt") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_brain_data(csv_path)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feature, y=TARGET, hue="Gender", ax=ax)
        ax.set_title(f"{feature} vs {TARGET}")
    fig.tight_layout()
    fig.savefig("scatter_mri.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sex in zip(axes, ["Female", "Male"]):
        corr = df[df["Gender"] == sex][FEATURES + [TARGET]].corr(method="pearson")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Корреляции Пирсона: {sex}")
    fig.tight_layout()
    fig.savefig("heatmap_corr.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    print(analyze_brain_correlations(BrainDataInput("brainsize.txt")))
    plot_brain_data()