# 📊 Loan Approval Prediction using Machine Learning
## 🧠 Project Overview

This project builds an end-to-end Machine Learning model to predict whether a loan application will be Approved or Not Approved based on applicant financial information.
The goal is to support data-driven loan approval decisions while balancing business risk and customer satisfaction.

---

## 🎯 Problem Statement

Loan approval is a critical decision for financial institutions.
Manual evaluation is time-consuming and subjective.

### Objective:
Build a machine learning model that predicts loan approval outcomes using applicant data such as:

- Income

- Credit Score

- Loan Amount

---

## 📁 Dataset Description

- Source: Kaggle

- Total Records: 1,000

- Target Variable: Approval Status

- 0 → Approved

- 1 → Not Approved

---

## 🔢 Features
- Feature	Description
- ApplicantID	Unique identifier
- Income	Annual income of applicant
- Credit Score	Creditworthiness score
- Loan Amount	Requested loan amount
- Approval Status	Target variable

---

## 🧹 Data Preprocessing

- Checked for missing values (None found)

- Removed duplicates (None found)

- Label encoded target variable

- Standardized numerical features using StandardScaler

- Split data into 80% training and 20% testing

---

## 📊 Exploratory Data Analysis (EDA)

- Analyzed distribution of approval status

- Reviewed statistical summary of numerical features

- Verified data consistency and scale differences

---

## 🤖 Models Implemented
### 1️⃣ Logistic Regression

- Used as a baseline model

- Applied class balancing

- Performance was limited due to linear assumptions

### 2️⃣ Random Forest Classifier ✅ (Final Model)

- Captures non-linear relationships

- Handles feature interactions well

- Used class_weight='balanced' to handle imbalance

- Provided better overall performance

---

## ⚙️ Model Evaluation Metrics

- Accuracy

- Precision

- Recall

- F1-score

- Confusion Matrix

- ROC-AUC Score

- Probability Threshold Tuning

---

## 🎯 Feature Importance Analysis

The Random Forest model identified the most influential features:

| Feature      | Importance |
| ------------ | ---------- |
| Loan Amount  | 34.68%     |
| Income       | 32.86%     |
| Credit Score | 32.46%     |

📌 Insight: Loan approval decisions are influenced almost equally by all three financial factors.

---

## 🏦 Business Insights

- Higher income and credit score increase approval probability

- Large loan amounts increase rejection risk

- Threshold tuning impacts bank risk vs customer approval rate

- Random Forest is better suited for financial decision modeling than linear models

---

## ✅ Final Model Selection

- Random Forest Classifier with default threshold (0.5) was selected as the final model due to:

- Better accuracy compared to Logistic Regression

- Balanced handling of approvals and rejections

- Strong feature importance interpretability

---

## 🧪 Technologies Used

- Python

- NumPy

- Pandas

- Matplotlib

- Seaborn

- Scikit-learn

- Kaggle Notebook Environment

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV

- SMOTE for advanced class imbalance handling

- Cross-validation

- Additional features (Employment status, Loan term)

- Deployment using Streamlit or Flask
