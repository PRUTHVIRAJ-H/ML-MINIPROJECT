# Review notes

## 1. Problem statement

The project predicts whether a Goodreads review is popular. Popularity is a
binary target: a review is positive when its likes and comments are more than
2% of all likes and comments received by that book.

This is a classification problem. The model is not predicting the exact
number of likes; it predicts the class `popular = 1` or `popular = 0`.

## 2. Dataset and preprocessing

The data comes from the UCSD Book Graph Goodreads review dataset. The
preparation pipeline:

1. Keeps reviews with at least one like or comment.
2. Keeps books with at least ten reviews and at least 60 total review
   likes/comments.
3. Keeps English reviews using the FastText language-identification model.
4. Computes each review's share of its book's total likes/comments.
5. Creates the binary popularity label using the 2% threshold.

The feature step creates:

- reviewer and timing information: `user_reviews`, `days_since_review`
- rating information: `user_rating`, `rating_diff`
- writing information: word count, average word/sentence length, and parts of
  speech percentages
- simple text signals: quotation mark presence and VADER sentiment

## 3. Method

The final implementation uses logistic regression because it is a strong,
simple baseline for binary classification and is easy to explain during the
demonstration. Features are standardized before training. The
`class_weight="balanced"` option prevents the more common class from
dominating the result without maintaining a separate undersampling script.

The data is split into 85% training and 15% testing using a fixed random seed
(`229`) and stratification. The test set is kept separate until evaluation.

## 4. How to explain the code

- `code/data_prep.py` is the data-cleaning stage.
- `code/features.py` converts reviews into numeric features that a model can use.
- `code/train_model.py` validates the input columns, splits the data, trains the
  pipeline, and prints evaluation results.
- `StandardScaler` puts numeric features on comparable scales.
- `LogisticRegression` estimates the probability that a review is popular.
- `classification_report` shows precision, recall, and F1-score.
- The confusion matrix shows correct and incorrect predictions by class.
- ROC-AUC measures how well the predicted probabilities separate popular and
  unpopular reviews; 0.5 is roughly random and 1.0 is perfect separation.

## 5. Demonstration flow

1. Show the problem statement and explain the 2% label.
2. Show a raw review and the corresponding engineered features.
3. Run the three commands in the README.
4. Explain the train/test split and why the test set is not used for training.
5. Show the classification report, confusion matrix, and ROC-AUC.
6. Mention that the model is intentionally kept interpretable and reproducible.

## 6. Likely questions

**Why logistic regression?**  
It is appropriate for a binary target, trains quickly, and gives a clear
baseline that is easier to interpret than a large neural network.

**Why use class balancing?**  
Popular reviews are less common. Balanced class weights make errors on both
classes matter during training without deleting training examples.

**Why is the split fixed?**  
A fixed seed makes the demonstration repeatable and allows another person to
reproduce the reported result.

**What could be improved?**  
Future work could compare several thresholds, use cross-validation, tune the
model, and add richer language features. Those changes are intentionally
outside the small, reliable demonstration version.

## 7. Conclusion

The project turns raw review metadata and text into measurable features and
uses them to classify review popularity. The final pipeline is short,
reproducible, and covers the complete path from preprocessing to evaluation.
