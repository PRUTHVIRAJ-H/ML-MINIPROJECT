# Goodreads Review Popularity

This machine-learning project predicts whether a Goodreads review is popular
or unpopular using review, reviewer, writing-style, and sentiment features.

## Run everything

From the repository root:

```bash
bash start.sh
```

The script creates the virtual environment, installs dependencies, downloads
the required data, prepares the features, trains the model, and evaluates it.
When it finishes, open [`reports/index.html`](./reports/index.html) in a
browser. The report is self-contained and puts the metric cards,
classification table, confusion matrix, ROC curve, and feature-influence graph
in one place.
