# House Price Prediction using Linear Regression

This project analyzes a housing dataset and applies **Linear Regression** to predict house prices. It includes preprocessing, outlier detection, data visualization, and training a regression model from scratch using gradient descent.

---

## 📂 Dataset

The dataset contains information about houses including:
- Square footage (`SqFt`)
- Number of bedrooms and bathrooms
- Number of offers
- Whether the house is made of brick
- Neighborhood category
- House price (target)

---

## 🔧 Preprocessing

- Filled missing values:
  - Used mean for `SqFt`
  - Used median for `Bedrooms`, `Bathrooms`, and `Offers`
- Encoded categorical variables (`Neighborhood`, `Brick`) using Label Encoding
- Performed outlier detection using:
  - IQR method
  - Z-score method

---

## 📊 Exploratory Data Analysis

- Scatter plots and regression lines for features vs. price
- Heatmap of feature correlations

---

## 🧠 Model

A **custom Linear Regression** model was implemented using NumPy, featuring:
- Feature normalization
- Gradient descent optimizer
- RMSE cost calculation

---

## ✅ Evaluation

- The model is trained on 90% of the dataset and evaluated on 10%
- Final RMSE on test set is printed for evaluation

---

## 📈 Visualization

- Loss trend during training
- Scatter and regression plots between features and price
- Correlation heatmap

---

## 🛠️ Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python house_price_prediction.py
