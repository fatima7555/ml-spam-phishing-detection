"""Email feature extraction: TF-IDF (bigrams) + 8 metadata features."""

import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

_CURRENCY_RE = re.compile(r"[$£€]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"(\+44|0)[\s\-]?[\d\s\-]{9,14}")


def _metadata(raw_texts) -> np.ndarray:
    rows = []
    for text in raw_texts:
        text = str(text)
        rows.append([
            len(text),
            len(text.split()),
            1 if _CURRENCY_RE.search(text) else 0,
            text.count("!"),
            sum(1 for c in text if c.isupper()) / max(len(text), 1),
            1 if _URL_RE.search(text) else 0,
            1 if _PHONE_RE.search(text) else 0,
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        ])
    return np.array(rows, dtype=float)


class EmailFeatureExtractor:
    def __init__(self, max_features: int = 10_000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.scaler = MinMaxScaler()

    def fit_transform(self, cleaned_texts, raw_texts):
        tfidf = self.vectorizer.fit_transform(cleaned_texts)
        meta = self.scaler.fit_transform(_metadata(raw_texts))
        return sp.hstack([tfidf, sp.csr_matrix(meta)])

    def transform(self, cleaned_texts, raw_texts):
        tfidf = self.vectorizer.transform(cleaned_texts)
        meta = self.scaler.transform(_metadata(raw_texts))
        return sp.hstack([tfidf, sp.csr_matrix(meta)])
