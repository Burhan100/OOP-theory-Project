# Student Study Hours Prediction using SVM

## 📄 Project Description

This project applies machine learning to predict student scores based on the number of study hours. We use **Support Vector Regression (SVR)** to create a predictive model. The project is structured in a beginner-friendly, object-oriented way.

---

## 📊 Dataset Information

* **Source** : [Kaggle - Student Study Hours](https://www.kaggle.com/datasets/himanshunakrani/student-study-hours)
* **File Name** : `student_scores.csv`
* **Features** :
* `Hours` – number of hours studied
* `Scores` – marks obtained

---

## 🧭 Project Workflow

### Step 1: Data Collection and Understanding

* Load the dataset using `pandas`
* Display the first few rows
* Check data types and missing values

### Step 2: Data Preprocessing and Analysis

* **Univariate Analysis** : Histogram plots to observe the distribution
* **Bivariate Analysis** : Correlation matrix and heatmap to check relationships

### Step 3: Data Splitting

* Use `train_test_split` to divide the dataset into training and testing sets

### Step 4: Model Training

* Create a pipeline with `StandardScaler` and `SVR`
* Train the model using the training set

### Step 5: Model Saving

* Save the trained model using Python’s `pickle` module
* The model is stored as `svm_study_model.pkl`

## 🗂️ Files in the Project

* `student_scores.csv` – Dataset
* `study_score_predictor.py` – Main script
* `svm_study_model.pkl` – Trained model
* `README.md` – Project explanation

---

## ✅ Summary

This project provides a simple but complete machine learning pipeline for regression tasks. It is ideal for beginners who want to understand how to load data, visualize it, train a model, and save it for future use.
