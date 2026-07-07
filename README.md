# Product Review Sentiment Classifier

Classifies a product/business review's text as **positive**, **negative**, or **neutral**.

## What it does

Given raw review text, the model predicts one of three sentiment classes. It's trained on real
customer reviews and their associated star ratings, using the star rating only as the source of
ground-truth labels — the model itself only ever sees the review text.

## Results

Two numbers are reported, from two separately trained models (same architecture, different
training data) — they aren't directly comparable, but both are standard things to report for a
3-class sentiment problem:

| Setup | Accuracy | Test set size |
|---|---|---|
| **3-class** (positive / neutral / negative) | **67.3%** | 3,000 reviews |
| **Binary** (positive / negative, neutral excluded) | **89.5%** | 2,000 reviews |

**3-class**, trained on 12,000 reviews (15,000 sampled, 80/20 split). Class-level performance:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.701 | 0.723 | 0.712 |
| Neutral | 0.578 | 0.546 | 0.562 |
| Positive | 0.731 | 0.749 | 0.740 |

Neutral is consistently the hardest class — 3-star reviews are inherently ambiguous/mixed in
sentiment, which is a well-known effect in 3-class sentiment analysis, not a modeling bug.
For reference, random guessing across 3 balanced classes would score ~33%.

**Binary**, trained on 8,000 reviews (10,000 sampled — the positive/negative subset of the same
data pool, neutral excluded entirely from both training and evaluation, 80/20 split):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.900 | 0.888 | 0.894 |
| Positive | 0.889 | 0.901 | 0.895 |

Removing the ambiguous neutral class recovers most of the accuracy lost to 3-way confusion —
consistent with how much of the 3-class model's error mass involves the neutral boundary.

See `confusion_matrix.png` (3-class) and `confusion_matrix_binary.png` (binary) for the full
confusion matrices, or run the notebook to regenerate them.

## Approach

**Data:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), a public
dataset of real business reviews with 1–5 star ratings. Star ratings are bucketed into 3 classes:
1–2★ → negative, 3★ → neutral, 4–5★ → positive. A stratified sample of 5,000 reviews per class
(15,000 total) is used for speed and clean class balance. The binary evaluation reuses this same
sampled pool, simply dropping the neutral rows and re-splitting/re-training on what's left.

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
