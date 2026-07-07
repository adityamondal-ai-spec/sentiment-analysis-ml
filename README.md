# Product Review Sentiment Classifier

Classifies a product/business review's text as **positive**, **negative**, or **neutral**.

## What it does

Given raw review text, the model predicts one of three sentiment classes. It's trained on real
customer reviews and their associated star ratings, using the star rating only as the source of
ground-truth labels — the model itself only ever sees the review text.

## Results

- **67.3% accuracy** on a held-out test set (3,000 reviews, never seen during training)
- Trained on **12,000 labeled reviews** (15,000 total sampled, 80/20 train/test split)
- Class-level performance (precision / recall / F1):

  | Class | Precision | Recall | F1 |
  |---|---|---|---|
  | Negative | 0.701 | 0.723 | 0.712 |
  | Neutral | 0.578 | 0.546 | 0.562 |
  | Positive | 0.731 | 0.749 | 0.740 |

  Neutral is consistently the hardest class — 3-star reviews are inherently ambiguous/mixed in
  sentiment, which is a well-known effect in 3-class sentiment analysis, not a modeling bug.
  For reference, random guessing across 3 balanced classes would score ~33%.

See `confusion_matrix.png` for the full confusion matrix, or run the notebook to regenerate it.

## Approach

**Data:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), a public
dataset of real business reviews with 1–5 star ratings. Star ratings are bucketed into 3 classes:
1–2★ → negative, 3★ → neutral, 4–5★ → positive. A stratified sample of 5,000 reviews per class
(15,000 total) is used for speed and clean class balance.

**Preprocessing** (via `TfidfVectorizer`):
- Lowercasing
- English stopword removal
- Unigrams + bigrams (`ngram_range=(1, 2)`), so short negations like *"not good"* are captured as
  a single feature instead of being lost
- Vocabulary capped at 30,000 terms; terms appearing in fewer than 2 documents are dropped

**Model:** a linear Support Vector Machine (`LinearSVC`). Linear SVMs are a standard, strong
baseline for high-dimensional sparse text features like TF-IDF and train in seconds at this
dataset size.

## Tech stack

Python · Scikit-learn · Pandas · NumPy · Matplotlib (for the confusion matrix plot)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib joblib requests jupyter

jupyter notebook notebook.ipynb
```

Run all cells top to bottom. The dataset (~23MB) downloads automatically on first run into
`data_raw/` (gitignored, not part of this repo) and the results above will reproduce exactly —
the random seed is fixed throughout.

To use the trained model directly without retraining:

```python
import joblib
model = joblib.load('sentiment_model.joblib')
model.predict(["This place was amazing, I'll definitely be back!"])
# ['positive']
```
