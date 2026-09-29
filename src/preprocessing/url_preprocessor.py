"""URL preprocessing: tokenize URL string for TF-IDF character n-grams."""

import re

_DELIMITERS_RE = re.compile(r"[/.\-_?=&@%+~:#]")
_SCHEME_RE = re.compile(r"^(https?|ftp)://", re.IGNORECASE)
_WWW_RE = re.compile(r"^www\d*\.", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Strip scheme and leading www. so input matches the training-data format."""
    url = str(url).strip().lower()
    url = _SCHEME_RE.sub("", url)
    url = _WWW_RE.sub("", url)
    return url


class URLPreprocessor:
    def tokenize_url(self, url: str) -> str:
        url = normalize_url(url)
        tokens = _DELIMITERS_RE.split(url)
        return " ".join(t for t in tokens if t)

    def transform(self, urls) -> list:
        return [self.tokenize_url(u) for u in urls]
