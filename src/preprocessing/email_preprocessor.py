"""Email text preprocessing: clean, tokenize, stem."""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for _resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource else f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"(\+44|0)[\s\-]?[\d\s\-]{9,14}")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def _clean(text: str) -> str:
    text = str(text)
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" URLTOKEN ", text)
    text = _EMAIL_RE.sub(" EMAILTOKEN ", text)
    text = _PHONE_RE.sub(" PHONETOKEN ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [_STEMMER.stem(t) for t in tokens if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


class EmailPreprocessor:
    def transform(self, texts) -> list:
        return [_clean(t) for t in texts]
