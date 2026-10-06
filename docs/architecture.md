# Project architecture

The project is a small, sequential machine-learning pipeline. Each stage
produces the input required by the next stage.

```text
Raw Goodreads data
        |
        v
code/data_prep.py
  - filters reviews and books
  - keeps English reviews
  - creates the popular/unpopular target
        |
        v
data/filtered_reviews.csv
        |
        v
code/features.py
  - creates rating, writing, grammar, and sentiment features
  - tokenizes and cleans review text
        |
        v
data/tokenized_reviews.csv
        |
        v
code/train_model.py
  - validates required columns
  - creates a stratified 85/15 train/test split
  - scales features
  - trains balanced logistic regression
  - prints evaluation metrics
        |
        v
Classification report, confusion matrix, and ROC-AUC
```

## Components

### Data preparation

`code/data_prep.py` reads the raw Goodreads JSON files and joins review data with
book data. It filters low-information records, identifies English reviews,
calculates each review's share of book-level engagement, and creates the
binary `popular` label.

### Feature engineering

`code/features.py` removes fields that should not be used directly, then creates
numeric features from ratings, review length, sentence structure, part-of-
speech ratios, quotations, and sentiment. The resulting CSV contains only the
target and model-ready features.

### Model and evaluation

`code/train_model.py` uses a scikit-learn pipeline. `StandardScaler` normalizes the
numeric inputs, and balanced `LogisticRegression` predicts the two classes.
The test set is held out until the final evaluation.

## Reproducibility

The train/test split and model use random seed `229`. Running the same
pipeline with the same input data produces a repeatable evaluation split.
