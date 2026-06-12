#  Twitter Sentiment Analysis — Apple & Google Products
### NLP End-to-End Binary Classification Project | Phase 4 | Moringa School

---

## Overview
This project builds a binary NLP classifier that automatically labels a tweet about Apple or Google products as either **Positive** or **Negative** based on its text content. The dataset is the CrowdFlower Twitter Brand Sentiment dataset (9,093 tweets) sourced from [data.world](https://data.world/crowdflower/brands-and-product-emotions), labelled by human raters.

---

##  Business Problem
Apple and Google product teams have no scalable, automated way to monitor whether public sentiment about their products on Twitter is positive or negative in real time. This classifier solves that by reading a tweet and instantly labelling it.

---

##  Objectives
1. Determine whether tweet text alone can reliably predict sentiment toward Apple and Google products.
2. Establish which classification algorithm performs best — a statistical model or a neural network.
3. Evaluate how TF-IDF feature engineering compares to Bag-of-Words in terms of model performance.

---

##  Data Understanding
- **9,093 tweets** across 3 columns — tweet text, product and sentiment label
- **4 sentiment classes** — Positive (32.7%), No Emotion (59.3%), Negative (6.3%), I can't tell (1.7%)
- **Binary subset** — 3,548 tweets retained (Positive: 2,978 | Negative: 570)
- **Class imbalance** — 84% Positive vs 16% Negative

---

##  Project Workflow
1. Data Loading & Understanding
2. Data Preparation — cleaning, filtering, encoding, NLTK preprocessing
3. EDA — Univariate & Bivariate Analysis
4. Train/Test Split (80/20 stratified)
5. Feature Engineering — BoW & TF-IDF
6. Baseline Modelling — 4 classifiers
7. GridSearchCV Tuning
8. Keras Neural Network
9. Final Evaluation
10. SHAP & LIME Model Interpretability
11. Conclusions & Recommendations

---

##  Results Summary

| Model | Weighted F1 |
|---|---|
| Naive Bayes (BoW) | 0.8430 |
| Logistic Regression (TF-IDF) | 0.8440 |
| Linear SVC (TF-IDF) | 0.8559 |
| Random Forest (TF-IDF) | 0.8559 |
| **Tuned Logistic Regression** | **0.8559** |
| Keras Neural Network | 0.8341 |

| Metric | Score | Target | Status |
|---|---|---|---|
| Weighted F1 | 0.8559 | ≥ 0.80 |  Met |
| Negative Recall | 0.4800 | ≥ 0.75 |  Not Met |
| ROC-AUC | 0.8624 | ≥ 0.85 |  Met |
| CV vs Test F1 | 0.0160 | ≤ 0.03 |  Met |

---

##  Key Findings
1. Tweet text alone reliably predicts sentiment — all 5 models exceeded F1 target of 0.80.
2. Statistical models outperform the Keras Neural Network on this dataset.
3. TF-IDF consistently outperforms Bag-of-Words across all models.
4. Class imbalance is the main challenge — Negative Recall of 0.48 falls below the 0.75 target.
5. SHAP reveals *link*, *ipad*, *iphone* and *new* are the most globally influential words.
6. LIME confirms product mentions combined with positive adjectives drive individual positive predictions.

---

##  Recommendations
1. Apply **SMOTE** to address class imbalance and improve Negative Recall.
2. Extend to **multiclass classification** including neutral tweets.
3. Explore **BERT or transformer-based models** for richer contextual embeddings.
4. Deploy the model as a **Flask or FastAPI endpoint** for real-time monitoring.

---

##  Technologies Used
Python | Pandas | NumPy | NLTK | Scikit-learn | TensorFlow/Keras | SHAP | LIME | Matplotlib | Seaborn | WordCloud

---

## ⚙️Setup Instructions
```bash
git clone https://github.com/nduvawinnie/twitter-sentiment-analysis.git
conda activate learn-env
pip install nltk lime shap tensorflow squarify wordcloud openpyxl
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
```

---

##  Author
**Winnie Wayua Nduva**
Payment Operations & Data Analyst |
[GitHub](https://github.com/nduvawinnie) | [LinkedIn](https://linkedin.com/in/winnie-nduva)
