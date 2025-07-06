# 🏠 Advanced House Price Prediction

This project predicts the sale prices of residential homes in Ames, Iowa, using advanced regression techniques. Leveraging a dataset with 79 explanatory variables that describe (almost) every aspect of a house, we train and evaluate multiple machine learning models to provide accurate price predictions.

---

## 📌 Project Overview

**Objective:**  
Predict the final sale price of homes in the Ames Housing dataset using various machine learning regression models.

**Dataset Source:**  
Ames Housing Dataset (Kaggle competition)

**Evaluation Metric:**  
Root Mean Squared Error (RMSE) between the **logarithm** of predicted and actual sale prices:
\[
\text{RMSE}_{\log} = \sqrt{\frac{1}{n} \sum (\log(\hat{y}) - \log(y))^2}
\]

---

## 📊 Dataset Summary

The dataset contains:
- **1460 training samples**, each with **81 columns** (80 features + 1 target)
- Features include:
  - Lot area, year built, basement quality, number of rooms, etc.
  - Both numerical and categorical features (including ordinal)

---

## 🧪 Workflow

1. **Exploratory Data Analysis (EDA)**
   - Understand distributions, missing values, and correlations.
   - Visualize features like `GrLivArea`, `SalePrice`, etc.

2. **Data Preprocessing**
   - Handling missing values intelligently (mode/median/None).
   - Label encoding for ordinal features (e.g., `ExterQual`, `BsmtQual`).
   - One-hot encoding for nominal categorical features.

3. **Feature Engineering**
   - Create new features (e.g., `TotalSF`, `Age`, `HasPool`).
   - Remove or transform skewed variables.

4. **Feature Scaling**
   - StandardScaler used for numerical features.

5. **Modeling**
   - Trained models:
     - `RandomForestRegressor`
     - `XGBRegressor` (XGBoost)
     - `LGBMRegressor` (LightGBM)
   - Used log-transformed `SalePrice` during training for stability.

6. **Evaluation**
   - RMSE and R² used for performance check.
   - Predictions transformed back using `np.expm1()` before submission.

---

## 🧠 Machine Learning Models

| Model              | RMSE (log scale) | R² Score |
|-------------------|------------------|----------|
| Random Forest      | ~34,000          | ~0.85    |
| XGBoost            | ~32,000          | ~0.86    |
| LightGBM           | ~31,944          | ~0.86    |

> 💡 Note: Values may vary depending on tuning and data split.

---
## 🛠️ Technologies Used

- Python (pandas, numpy, matplotlib, seaborn)
- Scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV or Optuna
- Ensemble/stacking for combining models
- Deployment as a web app (e.g., with Streamlit or Flask)
- Feature importance visualization

---

## 📈 Sample Output

- Feature importance plots
- Actual vs Predicted SalePrice scatter plots
- Log-RMSE and R² printed for each model

---

## 📬 Contact

If you have any questions or feedback, feel free to reach out.
