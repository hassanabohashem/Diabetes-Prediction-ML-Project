# Diabetes Prediction ML Project 🩺

## 🌟 Project Overview

This project implements a Machine Learning classification solution for the early detection of diabetes using the **Pima Indians Diabetes Dataset**.

The goal is to compare the performance of two popular classification algorithms—**Decision Tree** and **K-Nearest Neighbors (KNN)**—after applying essential data preprocessing steps.

***

## 🛠️ Methodology

The classification pipeline follows these key steps:

1.  **Data Loading:** Loading the `diabetes.csv` dataset.
2.  **Zero Imputation:** Replacing biologically implausible zero values (in columns like Glucose, BMI, BloodPressure, etc.) with the **mean** of the respective column.
3.  **Outlier Handling:** Removing extreme outliers in features like `SkinThickness` and `Insulin` using the **5th and 95th percentile** limits.
4.  **Feature Scaling (for KNN):** Applying **StandardScaler** to standardize feature values, which is crucial for distance-based algorithms like KNN.
5.  **Model Training:** Training Decision Tree (with `max_depth=4`) and KNN (with `n_neighbors=21`).
6.  **Evaluation:** Calculating Accuracy and plotting the **ROC Curve** and **Feature Importance** for the Decision Tree.

***

## 📊 Key Results

| Model | Test Accuracy | ROC AUC | Best Feature |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | (Enter your final Decision Tree Accuracy here, e.g., 0.76) | (Enter DT AUC here, e.g., 0.72) | Glucose |
| **KNN (Scaled)** | (Enter your final KNN Scaled Accuracy here, e.g., 0.71) | (Enter KNN AUC here, e.g., 0.67) | N/A |

*Note: The KNN model benefits significantly from feature scaling (StandardScaler).*

***

## 🚀 Getting Started

### Prerequisites

To run this project, you need Python and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pydotplus graphviz
