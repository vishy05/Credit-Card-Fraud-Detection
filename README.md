# Credit Card Fraud Detection 🚨💳

This project focuses on detecting fraudulent transactions using machine learning techniques. The dataset used is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains anonymized features and heavily imbalanced classes.

---

## 📊 Dataset Overview

- **Source:** European cardholders (Kaggle)
- **Size:** 284,807 transactions
- **Features:** 30 (28 anonymized PCA features + Time + Amount)
- **Target:** `Class` (0: Non-Fraud, 1: Fraud)
- **Imbalance:** Only 0.17% of the transactions are fraudulent

---

## 🚀 Project Workflow

1. **Data Loading & Exploration**
   - Checked for missing values
   - Analyzed class imbalance and feature distributions

2. **Preprocessing**
   - Feature scaling using `StandardScaler`
   - Addressed imbalance using `SMOTE`

3. **Modeling**
   - Trained multiple models:
     - Logistic Regression
     - Random Forest
     - XGBoost
   - Evaluated using precision, recall, F1-score, and ROC-AUC

4. **Model Evaluation**
   - Confusion matrix visualization
   - ROC Curve analysis
   - Feature importance

---

### Challenges Faced 
Class imbalance :

      -Skewed results if metrics like accuracy are used blindly.

      -ROC AUC and F1-score for class 1 used to avoid this issue.

Overfitting Risk:

      -Decision Trees and Random Forests overfit if not controlled with max_depth, min_samples_leaf.

Training Time:Random Forest was noticeably slow — can be due to default 100–500 estimators on a large dataset

---

## 📈 Performance Summary

| Model              | Precision | Recall | F1-Score | ROC-AUC(%) |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.83       | 0.64      | 0.72        | 0.957     |
|Decision Tree        | 0.75         | 0.74       | 0.75       |0.872      |
| Random Forest       | 0.81       | 0.83      | 0.82      | 0.973      |
| XGBoost             | 0.89     | 0.84     | 0.86      | 0.978      |

---

### Insights

Logistic Regression
Pros: Fast, interpretable, performs surprisingly well for a linear model.

Cons: Low recall (0.64) — meaning many fraud cases missed.

Verdict: Good benchmark but not acceptable for fraud detection due to high false negatives.

✅ Decision Tree
Pros: Better recall than logistic regression (0.74).

Cons: Still higher false positives; tends to overfit on training data; not stable.

Verdict: Decent for explainability, but outperformed by ensemble methods.

✅ Random Forest
Pros: Much better recall (0.83), low false negatives, balances well.

Cons: Slightly more false positives than XGBoost.

Verdict: Strong model; great trade-off between complexity and performance.

✅ XGBoost
Pros: Best across the board — highest precision (0.89), high recall (0.84), highest ROC AUC (0.978).

Cons: Training can be slower, needs tuning, but manageable with only 2–3 key features.

Verdict: ✅ Best performing model overall.

---

## 🛠 Technologies Used

- Python 3
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost
- Imbalanced-learn (`SMOTE`)

---

## 📌 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/CreditCardFraudDetection.git
   cd CreditCardFraudDetection
