"""Train and evaluate the project classifier.

The input is the feature file produced by features.py. The target is
``popular``: a review is popular when it receives more than 2% of its
book's review likes/comments.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the engineered data and validate the columns used by the model."""
    data = pd.read_csv(path).dropna()
    required = set(FEATURES + ["popular"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    return data[FEATURES].astype(float), data["popular"].astype(int)


def main() -> None:
    print("\n" + "=" * 72)
    print("STAGE 3/3: MODEL TRAINING AND EVALUATION")
    print("=" * 72)
    data_path = Path("data/tokenized_reviews.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} was not found. Run code/data_prep.py and "
            "code/features.py first."
        )

    features, target = load_data(data_path)
    print(f"Loaded {len(features):,} rows and {len(features.columns)} features.")
    print(
        "Class distribution: "
        f"{(target == 0).sum():,} unpopular, {(target == 1).sum():,} popular"
    )
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
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print("\n" + "-" * 72)
    print("EVALUATION RESULTS")
    print("-" * 72)
    print("Classification report (precision, recall, and F1-score):")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print("-" * 72)
    print("Interpretation: higher precision means fewer false positives;")
    print("higher recall means fewer missed popular reviews.")


if __name__ == "__main__":
    main()
