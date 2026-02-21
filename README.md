# 🏠 Delhi House Rent Price Prediction  
### End-to-End ML Project with Deployment

An end-to-end Machine Learning project that predicts house rental prices in Delhi using regression models. The project covers data preprocessing, EDA, feature engineering, model comparison, evaluation, and deployment using Streamlit.

---



## 📌 Problem Statement

Rental prices in Delhi depend on multiple factors such as:

- Location  
- Area (sq.ft)  
- Number of Bedrooms  
- Furnishing Status  
- Amenities  

This project builds a supervised regression pipeline to accurately predict rental prices.

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering  
4. Data Scaling (StandardScaler)  
5. Model Training  
6. Cross-Validation  
7. Performance Evaluation  
8. Model Saving (Joblib)  
9. Web App Deployment  

---

## 🤖 Models Implemented

- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Joblib  
- Streamlit  

---

# 🌐 Deployment Guide (Streamlit)

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/delhi-house-rent-price-prediction-ml.git
cd delhi-house-rent-price-prediction-ml
```

---

## 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📁 Recommended Project Structure

```
├── app.py
├── model.pkl
├── scaler.pkl
├── Indian_housing_Delhi_data.csv
├── requirements.txt
├── Delhi_House_Rent_Price_Prediction.ipynb
└── README.md
```

---

# 📦 requirements.txt

```
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

---

## 💡 Future Improvements

- Hyperparameter tuning  
- Feature importance visualization  
- Docker containerization  
- AWS/GCP deployment  
- CI/CD integration  

---
