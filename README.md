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

## 📈 Performance Summary

| Model              | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|-----------|--------|----------|---------|
| Logistic Regression | ~        | ~      | ~        | ~       |
|Decision Tree        |          |        |          |         |
| Random Forest       | ~        | ~      | ~        | ~       |
| XGBoost             | ~        | ~      | ~        | ~       |

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
