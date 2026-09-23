from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Project paths for data, outputs, and reports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "StudentsPerformance.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
PROCESSED_PATH = PROJECT_ROOT / "data" / "students_performance_processed.csv"
MISSING_PATH = REPORT_DIR / "missing_values.csv"
REPORT_PATH = REPORT_DIR / "findings.md"

# Name of Numerical Column
SCORE_COLUMNS = ["math_score", "reading_score", "writing_score"]
# Name of Categorical Column 
CATEGORY_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_education",
    "lunch",
    "test_preparation_course",]
# Mixing column for inspection
REQUIRED_COLUMNS = CATEGORY_COLUMNS + SCORE_COLUMNS

DISPLAY_NAMES = {
    "math_score": "Math",
    "reading_score": "Reading",
    "writing_score": "Writing",
    "overall_avg": "Overall average",}

VIVID_PALETTE = sns.color_palette("plasma", 3).as_hex()

# Data ingestion and preprocessing
def ingest_and_preprocess(
    input_path: Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read the CSV, standardize names, handle missing values, and add features."""

    df = pd.read_csv(input_path)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    df = df.rename(
        columns={
            "parental_level_of_education": "parental_education"
        }
    )

    missing_columns = set(REQUIRED_COLUMNS).difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The input dataset is empty.")

    missing_before = df.isna().sum()

    for column in CATEGORY_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

        if df[column].isna().any():
            df[column] = df[column].fillna(
                df[column].mode().iloc[0]
            )

    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if df[column].isna().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    df["overall_avg"] = df[SCORE_COLUMNS].mean(axis=1)

    return df, missing_before

# Utility functions for visualization and reporting
def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save each figure at exactly 800 x 600 pixels and 300 DPI."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# Create directories for outputs and reports
def ensure_directories() -> None:
    """Create output/report directories required for generated files."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_report_section(title: str, interpretation: str) -> None:
    """Write a report section for a visualization result."""
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(f"## {title}\n\n{interpretation}\n\n")


def show_figure() -> None:
    """Display a chart only in non-headless backends."""

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


def gender_math_reading_boxplot(df: pd.DataFrame) -> plt.Figure:
    """ Boxplot of math and reading scores by gender."""

    melted = df[["gender", "math_score", "reading_score"]].melt(
        id_vars="gender",
        var_name="subject",
        value_name="score",
    )

    melted["subject"] = melted["subject"].str.replace("_score", "", regex=False)
    melted["subject"] = melted["subject"].str.title()

    fig, ax = plt.subplots(figsize=(800 / 300, 600 / 300), dpi=300)

    sns.boxplot(
        data=melted,
        x="gender",
        y="score",
        hue="subject",
        palette=VIVID_PALETTE[:2],
        ax=ax,
    )

    ax.set_title("Boxplot of math and reading score by gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Score")
    ax.legend(
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        title_fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.5
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    return fig


def gender_interpretation(df: pd.DataFrame) -> str:
    """ Interpretation for the gender comparison figure."""

    gender_means = df.groupby("gender")[["math_score", "reading_score"]].mean()
    female_math = gender_means.loc["female", "math_score"]
    female_read = gender_means.loc["female", "reading_score"]
    male_math = gender_means.loc["male", "math_score"]
    male_read = gender_means.loc["male", "reading_score"]

    text = (
"The boxplots show that female students score a little higher than male students, especially in reading. "
f"Females have an average math score of {female_math:.1f} and reading score of {female_read:.1f}, "
f"while males have an average math score of {male_math:.1f} and reading score of {male_read:.1f}. "
"The difference between males and females is larger in reading than in math. "
"However, the scores overlap, which means that not every female student scores higher than every male student. "
"Overall, the data shows a small difference in scores between male and female students."
)

    return text


def lunch_type_and_average_performance(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of mean academic scores by lunch type."""

    lunch_means = (
        df.groupby("lunch")[["math_score", "reading_score", "writing_score"]]
        .mean()
        .reset_index()
    )

    melted = lunch_means.melt(
        id_vars="lunch",
        var_name="subject",
        value_name="mean_score",
    )
    melted["subject"] = melted["subject"].str.replace("_score", "", regex=False).str.title()

    fig, ax = plt.subplots(figsize=(800 / 300, 600 / 300), dpi=300)

    sns.barplot(
        data=melted,
        x="lunch",
        y="mean_score",
        hue="subject",
        palette=VIVID_PALETTE,
        ax=ax,
    )

    ax.set_title("Lunch Type and Average Performance")
    ax.set_xlabel("Lunch Type")
    ax.set_ylabel("Mean Score")
    fig.subplots_adjust(right=0.78)
    ax.legend(
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        title_fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.5,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    return fig


def lunch_type_interpretation(df: pd.DataFrame) -> str:
    """Interpretation for lunch-type comparison."""

    lunch_means = (
        df.groupby("lunch")[["math_score", "reading_score", "writing_score"]]
        .mean()
    )
    standard = lunch_means.loc["standard"]
    reduced = lunch_means.loc["free/reduced"]

    text = (
"The bar chart shows that students with a standard lunch have higher average scores in math, reading, and writing. "
f"Students with a standard lunch have average scores of {standard['math_score']:.1f} in math, "
f"{standard['reading_score']:.1f} in reading, and {standard['writing_score']:.1f} in writing. "
f"Students with a free/reduced lunch have average scores of {reduced['math_score']:.1f} in math, "
f"{reduced['reading_score']:.1f} in reading, and {reduced['writing_score']:.1f} in writing. "
"Overall, students with a standard lunch score higher on average in all three subjects."
)
    return text



def test_prep_math_boxplot(df: pd.DataFrame) -> plt.Figure:
    """Boxplot comparing math scores by test preparation status."""

    fig, ax = plt.subplots(figsize=(800 / 300, 600 / 300), dpi=300)

    sns.boxplot(
        data=df,
        x="test_preparation_course",
        y="math_score",
        hue="test_preparation_course",
        palette={"completed": VIVID_PALETTE[0], "none": VIVID_PALETTE[2]},
        dodge=False,
        legend=False,
        ax=ax,
    )

    ax.set_title("Math Scores by Test Preparation Course")
    ax.set_xlabel("Test Preparation Course")
    ax.set_ylabel("Math Score")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    return fig


def test_prep_math_interpretation(df: pd.DataFrame) -> str:
    """Interpretation for the test-prep chart."""

    means = df.groupby("test_preparation_course")["math_score"].mean()
    completed = means.get("completed", 0)
    none = means.get("none", 0)

    text = (
"The boxplot shows that students who completed the test preparation course have higher math scores on average. "
f"Students who completed the course have an average math score of {completed:.1f}, "
f"while students who did not complete it have an average score of {none:.1f}. "
"Overall, students who completed the test preparation course performed better in math in this dataset. "
"However, this does not mean that the test preparation course directly caused the higher scores."
)
    return text



def subject_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation between Math, reading and writing"""

    corr = df[SCORE_COLUMNS].corr()
    corr = corr.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=(800 / 300, 600 / 300), dpi=300)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="plasma",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title("Subject correlations")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Subject")

    return fig


def subject_correlation_interpretation(df: pd.DataFrame) -> str:
    """Interpretation for the subject-correlation."""

    corr = df[SCORE_COLUMNS].corr()
    math_reading = corr.loc["math_score", "reading_score"]
    math_writing = corr.loc["math_score", "writing_score"]
    reading_writing = corr.loc["reading_score", "writing_score"]

    text = (
"The heatmap shows a strong connection between math, reading, and writing scores. "
f"The correlation between math and reading is {math_reading:.2f}, "
f"math and writing is {math_writing:.2f}, and reading and writing is {reading_writing:.2f}. "
"Overall, students who score higher in one subject also tend to score higher in the other subjects."
)
    return text



def math_vs_reading_by_test_prep(df: pd.DataFrame) -> plt.Figure:
    """Math vs reading scatter plot with grouped trend lines by test preparation status."""

    fig, ax = plt.subplots(figsize=(800 / 300, 600 / 300), dpi=300)

    palette = {"completed": VIVID_PALETTE[0], "none": VIVID_PALETTE[2]}

    for group, group_df in df.groupby("test_preparation_course"):
        x = group_df["reading_score"]
        y = group_df["math_score"]
        sns.regplot(
            data=group_df,
            x="reading_score",
            y="math_score",
            scatter_kws={"s": 25, "alpha": 0.7, "color": palette.get(group, "#1F77B4")},
            line_kws={"color": palette.get(group, "#1F77B4"), "linewidth": 2},
            ax=ax,
            label=f"{group} (n={len(group_df)})",
            ci=None,
        )

    ax.set_title("Math vs reading with trend lines by test prep")
    ax.set_xlabel("Reading Score")
    ax.set_ylabel("Math Score")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(
        title="Test preparation course",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        title_fontsize=6,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.5,
    )

    return fig


def math_vs_reading_by_test_prep_interpretation(df: pd.DataFrame) -> str:
    """Interpretation for the math-reading relationship by test prep group."""

    groups = {}
    for group, group_df in df.groupby("test_preparation_course"):
        corr = group_df[["reading_score", "math_score"]].corr().loc["reading_score", "math_score"]
        groups[group] = corr

    completed_corr = groups.get("completed", 0.0)
    none_corr = groups.get("none", 0.0)

    text = (
"The scatter plot shows a strong connection between reading and math scores for both groups. "
f"Students who completed the test preparation course have a correlation of {completed_corr:.2f}, "
f"while students who did not complete it have a correlation of {none_corr:.2f}. "
"Overall, students with higher reading scores also tend to have higher math scores. "
"Students who completed the test preparation course also tend to have higher scores in both subjects."
)
    return text



def run_task(task_name: str, fig_fn, interpretation_fn, df: pd.DataFrame) -> None:
    """Generic execution step for each visualization task."""

    fig = fig_fn(df)
    show_figure()
    save_figure(fig, task_name)
    interpretation = interpretation_fn(df)
    write_report_section(task_name, interpretation)


if __name__ == "__main__":
    ensure_directories()
    REPORT_PATH.write_text("# Student Performance Findings\n\n", encoding="utf-8")

    df, missing_before = ingest_and_preprocess()
    df.to_csv(PROCESSED_PATH, index=False)
    missing_before.rename("missing_count").to_csv(MISSING_PATH, header=True)

    run_task("V1 - Gender differences in math vs reading", gender_math_reading_boxplot, gender_interpretation, df)
    run_task("V2 - Test prep impact on math", test_prep_math_boxplot, test_prep_math_interpretation, df)
    run_task("V3 - Lunch type and average performance", lunch_type_and_average_performance, lunch_type_interpretation, df)
    run_task("V4 - Subject correlations", subject_correlation_heatmap, subject_correlation_interpretation, df)
    run_task("V5 - Math vs reading with trend lines by test prep", math_vs_reading_by_test_prep, math_vs_reading_by_test_prep_interpretation, df)

    print(df.head())