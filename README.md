# 🏦 Home Credit Default Risk Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-LightGBM%20%7C%20SHAP%20%7C%20Pandas-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
Many people struggle to get loans due to insufficient or non-existent credit histories. This leads to reliable borrowers being rejected while risky borrowers might be approved.

**The Goal:** This project utilizes Machine Learning to predict repayment abilities (Default Risk), ensuring that capable applicants are approved while minimizing financial risk for the lender.

**The Challenge:**
* **Highly Imbalanced Data:** 92% of applicants are "Safe," making it hard to detect the 8% "Risky" ones.
* **Complex Data Structure:** Data is spread across 7 different tables (Bureau, Previous Applications, Installments, etc.).
* **Anomalies:** Employment data contained massive outliers (e.g., 1000 years of employment).

---

## 🏆 Key Results (Leaderboard)

| Rank | Model | ROC-AUC Score | Status |
| :--- | :--- | :--- | :--- |
| 🥇 | **LightGBM (Tuned)** | **0.7740** | **Final Model** |
| 🥈 | LightGBM (Baseline) | 0.7729 | Strong Contender |
| 🥉 | XGBoost | 0.7691 | |
| 4 | Random Forest | 0.7450 | Baseline |

> **Business Impact:** The final model achieves an AUC of **0.77**, significantly outperforming standard baselines. By correctly identifying high-risk applicants using behavioral history, this model can potentially save millions in default losses.

---

## ⚙️ Methodology

### 1. Data Cleaning & EDA
* **Anomaly Detection:** identified and fixed a "365243 Days Employed" (1000 years) error affecting 18% of the dataset.
* **Imputation:** Handled missing values using Median strategy to avoid skewing financial data.

### 2. Feature Engineering (The Secret Sauce)
The raw application data was not enough. I aggregated data from external sources to capture the **behavioral history** of applicants:
* **Bureau Data:** Aggregated credit history from other financial institutions (Avg Debt, Max Overdue).
* **Previous Applications:** Aggregated internal history with Home Credit (Refusal Rate, Avg Loan Amount).
* **Result:** Expanded feature space from **122 to 300+ columns**.

### 3. Modeling & Tuning
* **Algorithm:** LightGBM (Light Gradient Boosting Machine).
* **Handling Imbalance:** Used `class_weight='balanced'` to penalize false negatives.
* **Hyperparameter Tuning:** Optimized Learning Rate (0.02), Num Leaves (34), and Tree Depth using `RandomizedSearchCV`.

### 4. Interpretability (SHAP)
Using SHAP (SHapley Additive exPlanations), we opened the "Black Box" to understand *why* the model rejects certain people:
1.  **External Sources (EXT_SOURCE_2/3):** The strongest predictors of risk.
2.  **Previous Refusals:** Applicants rejected in the past are statistically much riskier.
3.  **Employment Stability:** Longer employment correlates strongly with repayment ability.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** LightGBM, XGBoost, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn, SHAP

---
