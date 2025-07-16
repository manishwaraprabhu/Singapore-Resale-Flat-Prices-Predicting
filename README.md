# 🏠 Singapore Resale Flat Prices Predicting

This project aims to develop a machine learning model and deploy it as a user-friendly web application to **predict the resale prices of flats in Singapore**. The predictive model is trained using historical resale flat transaction data and is designed to assist both potential **buyers** and **sellers** in estimating the market value of a flat based on key features.

## 📌 Project Objective

To build and deploy a machine learning model that:
- Analyzes historical HDB resale flat data
- Predicts the resale value of a flat based on user input
- Offers a web interface using **Streamlit** for ease of use

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy** – Data preprocessing
- **Scikit-learn** – Model building and evaluation
- **Streamlit** – Web app interface
- **VSCode** – Project development environment

## 📚 Data Source

The dataset was obtained from the official Singapore Housing Development Board (HDB) via [data.gov.sg](https://beta.data.gov.sg/collections/189/view).  
Five different datasets were provided by the team GUVI and downloaded for this project.

## 🧠 Project Workflow

### 1. **Environment Setup**
- Project developed in **VSCode**
- A **dedicated virtual environment** was created
- Required packages were installed using `pip`

### 2. **Data Collection & Preprocessing**
- Merged five datasets covering different timelines
- Performed:
  - Data cleaning
  - Handling of missing values
  - Standardization of column names

### 3. **Feature Selection**
- Selected key features for prediction:
  - `floor_area_sqm`
  - `flat_type`
  - `town`
  - `storey_range`
  - `flat_model`
  - `age_of_flat`
  - `remaining_lease_months`

### 4. **Feature Engineering**
- Applied **OneHotEncoding** to categorical columns:
  - `flat_type`, `town`, `storey_range`, `flat_model`
- Added encoded features back into the dataset with descriptive column names

### 5. **Model Selection & Training**
- Tried multiple regression models: **Linear Regression**, **Decision Trees**, and **Random Forest**
- **Random Forest Regressor** selected as the final model due to superior performance
- Trained the model on encoded categorical + numerical features
  - Features: `Town`, `Storey Range`, `Flat Model`, `Flat Type`, `Floor Area (sqm)`, `Remaining Lease (months)`, `Age of Flat (years)`
  - Target: `Resale Price`

### 6. **Model Evaluation**
- Evaluated using:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
- Random Forest model showed:
  - **Lowest MAE and RMSE**
  - **Highest R² score**, indicating strong predictive accuracy

### 7. **Streamlit Web App Development**
- Built a **Streamlit interface** to accept 7 user inputs:
  - `Town`, `Storey Range`, `Flat Model`, `Flat Type`, `Floor Area (sqm)`, `Remaining Lease (months)`, `Age of Flat (years)`
- Inputs are mapped to their encoded formats and passed to the trained model
- Predictions are made upon clicking the **"Predict Resale Price"** button

## 🖥️ Key Features of the Web App

- Interactive form to input flat details
- Real-time prediction using trained ML model
- Clean and intuitive UI built with Streamlit
- Fully functional and tested on live environment

## ✅ Results

- The model provides accurate price estimations for resale flats in Singapore.
- Helps buyers make informed decisions and helps sellers estimate property value.
- Demonstrates end-to-end ML pipeline from **data preprocessing to model deployment**.

## 🚀 Future Improvements

- Integrate model retraining with updated datasets
- Add data visualizations for trends by town/flat type
- Expand deployment to cloud platforms with persistent storage

## 📄 Deliverables

- ✅ Trained ML model
- ✅ Streamlit web application
- ✅ Encoders and preprocessing pipeline
- ✅ Deployment-ready app with documentation
