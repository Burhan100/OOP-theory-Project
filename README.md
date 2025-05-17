# Housing Price Prediction using SVM (Support Vector Machine)

This project predicts housing prices using a machine learning model called **Support Vector Regression (SVR)**. The code is written in Python using an object-oriented programming (OOP) approach to help beginners understand and structure real-world projects.

---

## 📌 Project Steps

### 1. Data Collection

- The dataset is loaded from a CSV file.
- The file used is typically from Kaggle, such as: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### 2. Data Processing

- **Univariate Analysis:** Describes each feature individually (e.g., mean, median, histograms).
- **Bivariate Analysis:** Shows relationships between features using a heatmap (correlation matrix).

### 3. Data Splitting

- The dataset is split into **training** and **testing** sets using `train_test_split` from scikit-learn.

### 4. Model Training

- An SVM model is trained using `SVR` (Support Vector Regression).
- Features are scaled using `StandardScaler`.
- The model is created as a pipeline to automate preprocessing and prediction.

### 5. Model Saving

- The trained model is saved using Python’s `pickle` module.
- This allows the model to be reused without retraining.

---

## 🛠️ Tech Stack

- Python 3.x
- pandas
- seaborn
- matplotlib
- scikit-learn
- pickle

---

## 📂 How to Run the Project

1. **Install required libraries:**

   ```bash
   pip install pandas seaborn matplotlib scikit-learn
   ```
