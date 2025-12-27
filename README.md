# Credit Card Fraud Detection Analysis

## Project Overview
This project aims to detect fraudulent credit card transactions using machine learning.  
The dataset contains anonymized transactions with features `V1` to `V28`, `Time`, and `Amount`, with a target variable `Class` (0 = Legit, 1 = Fraud).  
Fraud cases are extremely rare (~0.17% of all transactions), making it a **highly imbalanced dataset**.

---

## Repository Structure
- `Fraud_Analysis_Clean.ipynb`: Cleaned, fully reproducible notebook with all steps, visualizations, and model results.  
- `README.md`: Project explanation and results.  
- `images/` (optional): Folder for saved plots if needed in the future.

---

## Data Exploration & Preprocessing

### Dataset Info
- **Total entries:** 284,807  
- **Features:** 30 continuous features (`V1`-`V28`, `Amount`, `Time`)  
- **Target:** `Class` (Fraud/Legit)  

### Handling Class Imbalance
- The dataset is highly imbalanced:
  
| Class | Count  |
|-------|--------|
| 0     | 284,315 |
| 1     | 492    |

- **SMOTE (Synthetic Minority Oversampling Technique)** was used to balance the dataset by generating synthetic fraud samples, resulting in:

| Class | Count  |
|-------|--------|
| 0     | 284,315 |
| 1     | 284,315 |

- **Reasoning:** Balancing ensures models learn patterns of fraud instead of just predicting the majority class.

### Feature Engineering
- Converted `Time` into `Hour_of_Day` to analyze transaction time patterns.  
- Normalized `Amount` using scaling for better model performance.  
- No missing values were present.

---

## Model Training & Evaluation

### Train-Test Split
- **Train set:** 80%  
- **Test set:** 20%  
- **Balanced using SMOTE** only on the training set to avoid data leakage.

| Set       | Shape      | Class Distribution (Balanced) |
|-----------|-----------|------------------------------|
| X_train   | 454,904 x 41 | 0: 227,452, 1: 227,452      |
| X_test    | 113,726 x 41 | 0: 56,863, 1: 56,863        |

---

## Model Performance

### Logistic Regression
| Metric    | Class 0 | Class 1 | Weighted Avg |
|-----------|---------|---------|--------------|
| Precision | 1.00    | 0.10    | 0.99         |
| Recall    | 0.99    | 0.91    | 0.99         |
| F1-score  | 0.99    | 0.19    | 0.99         |
| Support   | 56,864  | 98      | 56,962       |

- **Observation:** Good at predicting legitimate transactions, poor at detecting fraud due to linearity limitations.

### Random Forest
| Metric    | Class 0 | Class 1 | Weighted Avg |
|-----------|---------|---------|--------------|
| Precision | 1.00    | 0.67    | 1.00         |
| Recall    | 1.00    | 0.87    | 1.00         |
| F1-score  | 1.00    | 0.76    | 1.00         |
| Support   | 56,864  | 98      | 56,962       |

- **Observation:** Captures fraud better due to non-linear tree-based structure.  
- High overall accuracy (~100%).

### XGBoost
| Metric    | Class 0 | Class 1 | Weighted Avg |
|-----------|---------|---------|--------------|
| Precision | 1.00    | 0.78    | 1.00         |
| Recall    | 1.00    | 0.87    | 1.00         |
| F1-score  | 1.00    | 0.82    | 1.00         |
| Support   | 56,864  | 98      | 56,962       |

- **Observation:** Best trade-off between precision and recall for fraud detection.  
- High interpretability with feature importance.

---

## Model Explainability
- **SHAP (SHapley Additive exPlanations)** was implemented for tree-based models (Random Forest & XGBoost).  
- **Insights:**
  - Features with the highest SHAP values contribute the most to predicting fraud.
  - Positive SHAP values increase fraud probability, negative decrease.  
  - Transparent feature contributions help auditors and banks understand model decisions.

**Note:** SHAP computations on large samples can be time-consuming; visualization is optional.

---

## Exploratory Data Analysis Summary
- Fraud transactions are concentrated in **low-amount ranges**, mostly at **odd hours**.  
- Features `V1`-`V28` (anonymized PCA components) are highly correlated with fraud patterns.  
- **Random Forest feature importance** highlighted the top predictors of fraud.

---

## Key Takeaways
1. Fraud detection requires balancing due to extreme class imbalance.  
2. Tree-based models (Random Forest, XGBoost) outperform Logistic Regression for rare event prediction.  
3. Model interpretability with SHAP is crucial for real-world deployment in banking.  
4. This project demonstrates a **complete pipeline** from data exploration, preprocessing, modeling, to interpretation.

---

## How to Use This Repository
1. Clone the repository.
2. Open `Fraud_Analysis_Clean.ipynb` in Jupyter Notebook.
3. Run all cells sequentially to reproduce:
   - Data preprocessing  
   - Exploratory Data Analysis (EDA)  
   - Model training & evaluation  
   - SHAP explainability (optional)  
4. Review the tables and plots in the notebook for insights.

---

## Technologies Used
- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- XGBoost
- SHAP

---

## License
This project is open-source for educational purposes.
