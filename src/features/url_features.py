"""URL feature extraction: TF-IDF char n-grams + 15 hand-crafted features."""

import re
import math
import numpy as np
import scipy.sparse as sp
from urllib.parse import urlparse, parse_qs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.url_preprocessor import normalize_url

_IP_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)
_SPECIAL_CHARS_RE = re.compile(r"[%=&~+]")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def _handcrafted(url: str) -> list:
    """Compute 15 hand-crafted URL features on the normalized URL form."""
    raw = str(url)
    url = normalize_url(raw)
    try:
        parsed = urlparse("http://" + url)
    except Exception:
        return [0.0] * 15

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    params = parse_qs(parsed.query)

    alpha_chars = [c for c in url if c.isalpha()]
    vowels = [c for c in alpha_chars if c.lower() in "aeiou"]

    return [
        len(url),                                               # 1 url_length
        url.count("."),                                         # 2 dot_count
        url.count("/"),                                         # 3 slash_count
        1 if _IP_RE.match(hostname) else 0,                    # 4 has_ip
        1 if parsed.scheme == "https" else 0,                  # 5 is_https
        1 if "@" in url else 0,                                # 6 has_at_symbol
        len(hostname),                                          # 7 domain_length
        max(hostname.count(".") - 1, 0),                       # 8 subdomain_count
        len(path),                                              # 9 path_length
        len(params),                                            # 10 num_params
        1 if parsed.port and parsed.port not in (80, 443) else 0,  # 11 has_port
        len(_SPECIAL_CHARS_RE.findall(url)),                    # 12 special_char_count
        sum(1 for c in url if c.isdigit()) / max(len(url), 1), # 13 digit_ratio
        len(vowels) / max(len(alpha_chars), 1),                # 14 vowel_ratio
        _shannon_entropy(url),                                  # 15 entropy
    ]


class URLFeatureExtractor:
    def __init__(self, max_features: int = 10_000):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
        )
        self.scaler = MinMaxScaler()

    def fit_transform(self, tokenized_urls, raw_urls):
        tfidf = self.vectorizer.fit_transform(tokenized_urls)
        hand = np.array([_handcrafted(u) for u in raw_urls], dtype=float)
        hand = self.scaler.fit_transform(hand)
        return sp.hstack([tfidf, sp.csr_matrix(hand)])

    def transform(self, tokenized_urls, raw_urls):
        tfidf = self.vectorizer.transform(tokenized_urls)
        hand = np.array([_handcrafted(u) for u in raw_urls], dtype=float)
        hand = self.scaler.transform(hand)
        return sp.hstack([tfidf, sp.csr_matrix(hand)])
