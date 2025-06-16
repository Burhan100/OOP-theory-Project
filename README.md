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

This project provides a simple but complete machine learning pipeline for regression tasks. It is ideal for beginners who want to understand how to load data, visualize it, train a model, and save it for futur<=> Stream Lit App Description  <=>

# Student Score Predictor Web App (Streamlit)

This is a machine learning web application built using **Streamlit** that allows users to:

1. **Log in securely**
2. **Upload a CSV file containing numeric data**
3. **Train a Linear Regression model**
4. **Predict scores (e.g., student marks) based on input values**
5. **Log out and return to the login page**

The project is designed to demonstrate a complete end-to-end ML workflow with a dynamic, user-friendly interface.

---

## 🧠 Purpose

This app serves as a **Student Marks Predictor**, where the predicted output (like `Scores`) is based on features such as `Hours` studied. However, the app is **generalized**, meaning it can handle **any dataset with numeric columns**, not just hours and scores.

The idea is to:

- Provide a secure entry point (login).
- Allow users to upload their dataset and define the target column.
- Train a machine learning model on the fly.
- Use the trained model to make predictions through manual input.
- Navigate through pages in a controlled manner (like a real application).

---

## 🚀 Features

### ✅ 1. Login Page

- Simple authentication system.
- Prevents access to other pages until the user logs in.

### 📤 2. Upload & Train Model Page

- Upload a `.csv` file containing numeric columns.
- Automatically identifies numeric columns from the dataset.
- Select a **target column** (value you want to predict).
- Trains a **Linear Regression model** on the selected data.
- Saves the model and feature columns in session state for later use.

### 📈 3. Prediction Page

- Dynamically generates input fields for each feature column.
- Takes user input, formats it into a DataFrame, and makes a prediction using the trained model.
- Displays the predicted result (e.g., predicted score).
- Includes a **Logout** button to return to the login page and clear the session.

### 🔒 4. Session Management

- Navigation is controlled using `st.session_state`.
- Users can only access one page at a time.
- If not logged in, other pages are blocked.
- Logout resets the session state and returns to the login page.

---

## 🛠 How to Use

### 1. Clone or Download the Project
