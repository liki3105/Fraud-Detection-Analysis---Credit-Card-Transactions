# Fraud Detection Analysis: Credit Card Transactions

## Project Goal
Detect fraudulent credit card transactions using machine learning models with detailed feature engineering, handling class imbalance, and interpretable results.

---

## 1. Dataset Overview
- Source: Kaggle "Credit Card Fraud Detection"
- Size: 284,807 transactions
- Features:
  - 28 anonymized principal components (`V1`–`V28`)
  - `Amount` of transaction
  - `Time` (seconds from first transaction)
  - Target `Class` (0 = Legit, 1 = Fraud)

### Dataset Summary

| Feature | Type | Non-Null Count |
|---------|------|----------------|
| Time    | float | 284,807 |
| V1      | float | 284,807 |
| V2      | float | 284,807 |
| ...     | ...   | ...     |
| V28     | float | 284,807 |
| Amount  | float | 284,807 |
| Class   | int   | 284,807 |

**Original Class Distribution**

| Class | Count |
|-------|-------|
| 0     | 284,315 |
| 1     | 492 |

---

## 2. Feature Engineering
- Created time-based features from `Time`:
  - `Time_Hours`, `Time_Days`, `Hour_of_Day`, `Minute_of_Hour`, `Second_of_Minute`
- Captures temporal patterns for fraud detection

---

## 3. Handling Class Imbalance
- Used **SMOTE** to balance classes:

| Class | Count |
|-------|-------|
| 0     | 284,315 |
| 1     | 284,315 |

- Train/Test split (80/20):

| Set | Shape | Class 0 | Class 1 |
|-----|-------|---------|---------|
| X_train | (454,904, 41) | 227,452 | 227,452 |
| X_test  | (113,726, 41) | 56,863  | 56,863  |

---

## 4. Modeling & Results

### 4.1 Logistic Regression
| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Precision | 1.00 | 0.10 |
| Recall    | 0.99 | 0.91 |
| F1-score  | 0.99 | 0.19 |
| Support   | 56,864 | 98 |

**Confusion Matrix**

|       | Pred 0 | Pred 1 |
|-------|--------|--------|
| True 0| 56,102 | 762    |
| True 1| 9      | 89     |

---

### 4.2 Random Forest
| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Precision | 1.00 | 0.67 |
| Recall    | 1.00 | 0.87 |
| F1-score  | 1.00 | 0.76 |
| Support   | 56,864 | 98 |

**Confusion Matrix**

|       | Pred 0 | Pred 1 |
|-------|--------|--------|
| True 0| 56,823 | 41     |
| True 1| 13     | 85     |

---

### 4.3 XGBoost
| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Precision | 1.00 | 0.78 |
| Recall    | 1.00 | 0.87 |
| F1-score  | 1.00 | 0.82 |
| Support   | 56,864 | 98 |

**Confusion Matrix**

|       | Pred 0 | Pred 1 |
|-------|--------|--------|
| True 0| 56,840 | 24     |
| True 1| 13     | 85     |

---

## 5. Visualizations
- Class distribution (bar chart)  
- Transaction `Amount` distribution (histogram)  
- Confusion matrices  
- Feature importance (Random Forest & XGBoost)  

*SHAP plots were attempted but skipped due to runtime issues. Feature importance used instead.*

---

## 6. Challenges & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Logistic Regression convergence warning | `max_iter` too low | Increased `max_iter=1000` |
| Random Forest slow training | Large dataset | Limited estimators, sampled for plots |
| SHAP errors | Version mismatch / multi-output | Skipped, used feature importance |
| Kernel crashes / long runtime | Large dataset | Sampled subset for plotting; set `random_state` |

---

## 7. Conclusion
- Detected fraudulent transactions using **Random Forest & XGBoost**
- Engineered time-based features
- Handled class imbalance with **SMOTE**
- Validated with precision, recall, F1, and confusion matrices
- Feature importance highlights key drivers of fraud

---

## 8. Next Steps
- Deploy **XGBoost** for real-time monitoring
- Re-integrate SHAP once runtime issues are resolved
- Explore unsupervised anomaly detection for unseen fraud patterns

---

## 9. File Organization
Fraud_Analysis_Clean.ipynb # Clean, final notebook
README.md # Documentation


---

## 10. Tools & Libraries
- Python 3.10  
- Pandas, Numpy, Matplotlib, Seaborn  
- Scikit-learn (Logistic Regression, Random Forest)  
- XGBoost  
- Imbalanced-learn (SMOTE)
