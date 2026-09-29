# PhishGuard — Spam & Phishing Detection System

ML Lab Project · Binary classifier for detecting malicious emails and phishing URLs.

Two independent classifiers (email spam + URL phishing) unified under a single Flask web interface, with NLP preprocessing, TF-IDF vectorization, hyperparameter optimization, and Docker containerization.

---

## Stack

`Python 3.11+` · `scikit-learn` · `XGBoost` · `NLTK` · `Flask` · `Bootstrap 5` · `Docker`

---

## Project Structure

```
ML_Final_Project/
├── data/
│   ├── download_data.py          # downloads SMS Spam + Phishing URL datasets
│   └── raw/                       # downloaded CSV files
├── src/
│   ├── preprocessing/             # email + URL cleaning
│   ├── features/                  # TF-IDF + hand-crafted features
│   └── models/model_selector.py   # 5 models + 3-tier selection logic
├── train_email.py                 # email training pipeline
├── train_url.py                   # URL training pipeline
├── evaluate.py                    # full evaluation report
├── webapp/                        # Flask web application
│   ├── app.py
│   ├── templates/index.html
│   └── static/style.css
├── saved_models/                  # trained model artefacts (.pkl)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Option 1 · Local Python

```powershell
# 1. install dependencies
pip install -r requirements.txt

# 2. download NLTK corpora
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# 3. download datasets
python data/download_data.py

# 4. train both classifiers (takes ~5–10 min total)
python train_email.py
python train_url.py

# 5. run full evaluation report
python evaluate.py

# 6. start web app
python webapp/app.py
# open http://localhost:5000
```

### Option 2 · Docker

```powershell
docker-compose build      # ~15-20 min — datasets download & models train inside
docker-compose up -d
# open http://localhost:5000
```

---

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| SMS Spam Collection | UCI ML Repository | 5,574 messages (~13% spam) | Email spam classifier |
| Phishing URLs (faizann24) | GitHub public dataset | 30k URLs (50/50 stratified sample, augmented with 100 curated legitimate URLs) | URL phishing classifier |

Both datasets download automatically via `data/download_data.py`. An embedded fallback dataset is included for offline use.

---

## Models Compared (Both Classifiers)

| Model | Library |
|---|---|
| Multinomial Naive Bayes | scikit-learn |
| Logistic Regression | scikit-learn |
| Linear SVM (Calibrated) | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |

All 5 models are tuned with **`GridSearchCV` (cv=5, scoring='f1_macro')**.

---

## Model Selection Logic (Viva-Defensible)

A 3-tier explicit criterion is applied (documented in `src/models/model_selector.py`):

```
Tier 1: Highest macro F1 on 5-fold CV       ← primary metric (handles class imbalance)
Tier 2: Highest recall on malicious class   ← security domain (false negatives are costly)
Tier 3: Fastest inference time              ← deployment efficiency
Disqualify: any model with test F1 < 0.90
```

The reason for choosing **macro F1** over accuracy: SMS Spam is 87% ham / 13% spam — accuracy can be inflated by always predicting the majority class. Macro F1 equally weights both classes.

The reason for **recall on malicious class**: in security applications, missing a malicious item (false negative) is more harmful than flagging a legitimate one (false positive).

---

## Features

### Email (TF-IDF + 8 metadata features)
- `TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)` on cleaned text
- Metadata: text_length, word_count, has_currency_symbol, exclamation_count, capitals_ratio, has_url, has_phone, digit_ratio

### URL (TF-IDF char n-grams + 15 hand-crafted features)
- `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=10000)` on tokenized URLs
- Hand-crafted: url_length, dot_count, slash_count, has_ip, is_https, has_at_symbol, domain_length, subdomain_count, path_length, num_params, has_port, special_char_count, digit_ratio, vowel_ratio, **Shannon entropy**

---

## Web Application

Two-tab Bootstrap 5 UI:
- **Email Spam Checker** — paste email content
- **URL Phishing Checker** — paste a URL

Each prediction returns a label (SPAM / LEGITIMATE / PHISHING), a confidence percentage, and a colored badge.

### REST API

| Endpoint | Method | Body |
|---|---|---|
| `/` | GET | renders index.html |
| `/predict/email` | POST | `{"text": "..."}` |
| `/predict/url` | POST | `{"url": "..."}` |
| `/health` | GET | model loading status |

Response shape:
```json
{ "label": "SPAM", "is_malicious": true, "confidence": 95.5 }
```

---

## Verification Tests

```powershell
# spam check
curl -X POST http://localhost:5000/predict/email -H "Content-Type: application/json" -d '{\"text\":\"Congratulations! You won a free prize - call now to claim!\"}'

# phishing URL check
curl -X POST http://localhost:5000/predict/url -H "Content-Type: application/json" -d '{\"url\":\"paypal-security-alert.phishing-site.com/login\"}'

# legitimate URL check
curl -X POST http://localhost:5000/predict/url -H "Content-Type: application/json" -d '{\"url\":\"wikipedia.org/wiki/Machine_learning\"}'
```

---

## License & Credits

Built as a Machine Learning lab project. Datasets credit:
- UCI ML Repository (SMS Spam Collection)
- faizann24/Using-machine-learning-to-detect-malicious-URLs (GitHub)
