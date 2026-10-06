# Text-Based Prediction of Book Review Popularity

Machine Learning Mini-Project — UE24CS352A

## Run the complete project with one command

```bash
bash start.sh
```

This one command creates the virtual environment(for python), installs all requirements,
downloads the required NLTK resources(the book reviews data), prepares the data, creates the
features, trains the model, and creates one complete HTML results report.

After the command finishes, open:

```text
reports/index.html
```

The report includes the headline metrics, full
classification table, confusion-matrix graph, ROC curve, and feature-influence
graph.

The script creates these local data files automatically:

```text
data/goodreads_reviews_dedup.json
data/goodreads_books.json
lid.176.bin
```

## 1. Project overview

This project predicts whether a Goodreads book review is **popular** or
**unpopular**. A review is labelled popular when it receives more than 2% of
the total likes and comments received by reviews of the same book.

The project is a binary classification problem. It does not predict the exact
number of likes; it predicts one of two classes:

- `1`: popular review
- `0`: unpopular review

## 2. Problem statement

Review popularity depends on several factors, including the reviewer's
activity, rating compared with the book's average rating, review length,
writing style, and sentiment. The goal is to use these measurable properties
to classify reviews into popular and unpopular categories.

## 3. Dataset

The project uses the Goodreads review and book data from the UCSD Book Graph
dataset. To keep the one-command demonstration practical, `start.sh`
automatically downloads the official **poetry subset** on the first run. The
raw files used locally are:

```text
data/goodreads_reviews_dedup.json
data/goodreads_books.json
lid.176.bin
```

The FastText model `lid.176.bin` is used to keep English-language reviews.
Large raw files and generated data are excluded from the repository.
The first startup downloads approximately 350 MB of compressed data and model
files, extracts the two JSON files, and reuses them on later runs.

Official sources:

- Original project code: <https://github.com/bridgetdaly/goodreads_ML>
- UCSD Book Graph dataset:
  <https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html>
- FastText language-identification model:
  <https://fasttext.cc/docs/en/language-identification.html>

See [DATA_SOURCES.md](./DATA_SOURCES.md) and [data.md](./data.md)
for the exact file locations.

### Data preparation

`data_prep.py`:

1. Reads the raw review and book JSON files.
2. Keeps reviews with at least one like or comment.

3. Keeps books with at least ten reviews.
4. Keeps books with at least 60 total review likes/comments.
5. Filters for English reviews.

6. Calculates each review's share of its book's engagement.
7. Creates the `popular` target using the 2% threshold.

The output is `data/filtered_reviews.csv`.

## 4. Feature engineering

`features.py` converts the cleaned data into model features:

- `user_reviews`: number of reviews written by the user
- `days_since_review`: age of the review
- `user_rating`: rating given by the user
- `rating_diff`: user rating minus the book's average rating
- `num_words`: number of alphabetic words
- `avg_word_len`: average word length
- `avg_sent_len`: average sentence length
- `pct_verbs`, `pct_nouns`, `pct_adj`: parts-of-speech percentages
- `quote`: whether the review contains a quotation
- `sentiment`: VADER sentiment score

The output is `data/tokenized_reviews.csv`.

## 5. Machine-learning approach

The final model is a balanced logistic-regression classifier.

1. The data is divided into 85% training and 15% testing data.
2. The split is stratified so both classes are represented consistently.
3. Features are standardized with `StandardScaler`.

4. Logistic regression predicts the probability of popularity.
5. `class_weight="balanced"` gives appropriate importance to both classes.

6. The model is evaluated only on the held-out test set.

Logistic regression was selected because it is appropriate for a binary
target, trains quickly, is reproducible, and is easy to explain during the
review. The fixed random seed is `229`.

### What logistic regression is learning

This is **supervised binary classification**: the training data contains both
the review features and the correct answer (`0` for unpopular, `1` for
popular). Logistic regression is a **parametric, discriminative** model. It
learns one weight for each feature and a bias:

```text
z = b + w1*x1 + w2*x2 + ... + wn*xn
p(popular) = 1 / (1 + e^(-z))
```

The sigmoid function converts the linear score into a probability between 0
and 1. During training, scikit-learn chooses the weights that minimize binary
cross-entropy (also called log loss), together with L2 regularization from the
default solver. A probability of at least 0.5 is predicted as popular. The
decision boundary is linear; the model is not learning a collection of rules
or memorizing individual reviews.

`StandardScaler` is applied before the logistic-regression step so features
with different units can contribute fairly. `class_weight="balanced"` gives
the less frequent popular class appropriate importance.

### Training versus testing

- **Training:** the model reads the 85% training rows, adjusts its weights,
  and minimizes the loss. Training time is the time spent in `model.fit(...)`.
- **Testing:** the model receives the untouched 15% test rows and applies the
  learned weights to make predictions. Testing time is the time spent in
  `predict(...)` and `predict_proba(...)`; no weights are changed.
- **Evaluation:** metrics such as precision, recall, F1-score, confusion
  matrix, and ROC-AUC compare predictions with the known test labels. This is
  separate from testing inference time.

The HTML report displays all three timings. For each unseen review, testing
responds with both a predicted class and a probability of being popular.
Testing is normally faster than training because prediction only performs the
learned calculation; it does not search for model parameters. The report's
metric-calculation time is separate from prediction time because it compares
the predictions with the known test labels.

## 6. Project architecture

```text
Raw Goodreads JSON + FastText model
                 |
                 v
          data_prep.py
                 |
                 v
      data/filtered_reviews.csv
                 |
                 v
            features.py
                 |
                 v
      data/tokenized_reviews.csv
                 |
                 v
          train_model.py
                 |
                 v
       reports/index.html
```

See [architecture.md](./architecture.md) for the detailed architecture
explanation.

## 7. Installation and one-command startup

The complete project can be installed, supplied with the official subset, and
executed with this single command:

```bash
bash start.sh
```

The script:

1. Checks that Python 3 and `curl` are installed.
2. Creates or reuses the `.venv` virtual environment.
3. Installs all packages from `requirements.txt`.
4. Downloads the required NLTK resources.
5. Downloads the official poetry subset and FastText model if missing.
6. Runs data preparation, feature engineering, and model evaluation.
7. Prints clear progress messages and final metrics.

For full setup details and troubleshooting, see [start.md](./start.md).

## 8. Manual installation and execution

If you prefer to run each step yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download NLTK resources once:

```bash
python -c "import nltk; [nltk.download(x) for x in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'vader_lexicon', 'stopwords', 'wordnet']]"
```

Then run:

```bash
python code/data_prep.py
python code/features.py
python code/train_model.py
```

The final command writes `reports/index.html`.

## 9. Output and evaluation

`train_model.py` creates:

- metric cards for accuracy, ROC-AUC, precision, recall, and F1-score
- a precision/recall/F1 classification table
- a confusion matrix graph
- a ROC curve graph
- a feature-influence graph based on model coefficients
- training, testing, and evaluation timings
- a plain-language explanation of the learning type, sigmoid function, and
  train/test difference

It also prints a compact summary for command-line users. The generated HTML
contains embedded images, so it can be copied or opened without any other
project files.

The pipeline was verified successfully with `bash start.sh` using the official
poetry subset:

```text
Usable reviews: 9,663
Predictor features: 12
Accuracy: 0.72
ROC-AUC: 0.732
Popular-review precision: 0.49
Popular-review recall: 0.55
Popular-review F1-score: 0.52
```

These results are for the poetry subset downloaded by `start.sh`, not the
complete multi-gigabyte Goodreads archive.

The classification report measures performance for both popular and
unpopular reviews. The confusion matrix shows correct and incorrect
predictions. ROC-AUC measures how well the predicted probabilities separate
the two classes.

## 10. Review explanation

For the demonstration, explain the project in this order:

1. We solve a binary classification problem: predicting review popularity.
2. Popularity is defined as receiving more than 2% of a book's total review
   likes/comments.
3. We clean the Goodreads data and filter non-English and low-information
   records.
4. We engineer rating, reviewer, writing-style, and sentiment features.
5. We standardize the features and train balanced logistic regression.
6. We evaluate on a separate 15% test set using precision, recall, F1-score,
   confusion matrix, and ROC-AUC.
7. The pipeline is intentionally simple, reproducible, and interpretable.

## 11. Repository contents

- [data_prep.py](../code/data_prep.py): raw-data filtering and target creation
- [features.py](../code/features.py): feature engineering
- [train_model.py](../code/train_model.py): model training and evaluation
- [start.sh](../start.sh): one-command installation and execution
- [architecture.md](./architecture.md): architecture and data flow
- [start.md](./start.md): startup instructions and troubleshooting
- [notes.md](./notes.md): review notes and likely questions
- [requirements.txt](./requirements.txt): Python dependencies
- [DATA_SOURCES.md](./DATA_SOURCES.md): dataset and code sources
- [data.md](./data.md): required input-file locations

## 12. Deliverables

The repository contains the source code and setup documentation required for
the project.