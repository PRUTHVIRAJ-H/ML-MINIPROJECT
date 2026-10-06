"""Train the classifier and create one self-contained evaluation report."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 229
FEATURES = [
    "user_reviews",
    "days_since_review",
    "user_rating",
    "rating_diff",
    "num_words",
    "avg_word_len",
    "avg_sent_len",
    "pct_verbs",
    "pct_nouns",
    "pct_adj",
    "quote",
    "sentiment",
]
REPORT_PATH = Path("reports/index.html")


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load engineered data and validate the model columns."""
    data = pd.read_csv(path).dropna()
    required = set(FEATURES + ["popular"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return data[FEATURES].astype(float), data["popular"].astype(int)


def image_data(fig: plt.Figure) -> str:
    """Convert a plot to an embedded PNG so the report has no dependencies."""
    output = BytesIO()
    fig.savefig(output, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode("ascii")


def make_confusion_plot(matrix: list[list[int]]) -> str:
    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Unpopular", "Popular"],
        yticklabels=["Unpopular", "Popular"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return image_data(fig)


def make_roc_plot(actual: pd.Series, probabilities: object) -> str:
    false_positive, true_positive, _ = roc_curve(actual, probabilities)
    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    axis.plot(false_positive, true_positive, label="Model")
    axis.plot([0, 1], [0, 1], "--", color="#888888", label="Random")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="ROC curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    return image_data(fig)


def make_coefficients_plot(model: object) -> str:
    coefficients = model[-1].coef_[0]
    values = pd.Series(coefficients, index=FEATURES).sort_values()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    values.plot.barh(ax=axis, color=["#d95f02" if value < 0 else "#1b9e77" for value in values])
    axis.set(xlabel="Logistic-regression coefficient", title="Feature influence")
    return image_data(fig)


def metric_cards(report: dict[str, dict[str, float]], auc: float) -> str:
    popular = report["1"]
    return "".join(
        f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>'
        for label, value in [
            ("Accuracy", f"{report['accuracy']:.2f}"),
            ("ROC-AUC", f"{auc:.3f}"),
            ("Popular precision", f"{popular['precision']:.2f}"),
            ("Popular recall", f"{popular['recall']:.2f}"),
            ("Popular F1", f"{popular['f1-score']:.2f}"),
        ]
    )


def make_report(
    report: dict[str, dict[str, float]],
    matrix: list[list[int]],
    auc: float,
    rows: int,
    train_rows: int,
    test_rows: int,
    confusion_image: str,
    roc_image: str,
    coefficient_image: str,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table_rows = "".join(
        f"<tr><th>{label.title()}</th><td>{values['precision']:.2f}</td>"
        f"<td>{values['recall']:.2f}</td><td>{values['f1-score']:.2f}</td>"
        f"<td>{values['support']:.0f}</td></tr>"
        for label, values in report.items()
        if label in {"0", "1"}
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goodreads review popularity results</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; background: #f4f7fb; color: #172033; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
h1 {{ margin-bottom: .25rem; }} .muted {{ color: #5d687a; }}
.metrics, .plots {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin: 1.5rem 0; }}
.metric, .card {{ background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 12px #17203314; }}
.metric strong, .metric span {{ display: block; }} .metric strong {{ font-size: 1.8rem; color: #2457a6; }}
.metric span {{ color: #5d687a; font-size: .9rem; }} .card {{ overflow: auto; }}
.card img {{ display: block; max-width: 100%; height: auto; margin: auto; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ padding: .7rem; border-bottom: 1px solid #e3e8ef; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }} footer {{ margin-top: 2rem; color: #5d687a; font-size: .9rem; }}
</style>
</head>
<body>
<h1>Goodreads review popularity results</h1>
<p class="muted">Generated by the complete pipeline. Popular means more than 2% of a book's review engagement.</p>
<section class="metrics">{metric_cards(report, auc)}</section>
<div class="card"><h2>Classification report</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>
<tbody>{table_rows}</tbody></table></div>
<section class="plots">
<div class="card"><img alt="Confusion matrix" src="data:image/png;base64,{confusion_image}"></div>
<div class="card"><img alt="ROC curve" src="data:image/png;base64,{roc_image}"></div>
</section>
<div class="card"><img alt="Feature influence chart" src="data:image/png;base64,{coefficient_image}"></div>
<footer>{rows:,} usable rows &middot; {train_rows:,} training rows &middot; {test_rows:,} test rows &middot; {len(FEATURES)} features</footer>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    data_path = Path("data/tokenized_reviews.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} was not found. Run code/data_prep.py and code/features.py first."
        )

    features, target = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions).tolist()
    auc = roc_auc_score(y_test, probabilities)
    make_report(
        report,
        matrix,
        auc,
        len(features),
        len(X_train),
        len(X_test),
        make_confusion_plot(matrix),
        make_roc_plot(y_test, probabilities),
        make_coefficients_plot(model),
    )
    print(f"Usable reviews: {len(features):,}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Confusion matrix:\n{matrix}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Saved complete report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
