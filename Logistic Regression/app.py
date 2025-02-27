import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)
    
# Load the saved scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
    
# Default mean values (to replace nulls in the uploaded data)
default_means = {
    "Age": 29,
    "Fare": 32.2,
    "Pclass": 2,
    "Sex": "Unknown",
    "SibSp": 0,
    "Parch": 0,
    "Dependants": 0,
    "Embarked":"Unknown",
    "Sex_male":0,
    "Embarked_C":0,
    "Embarked_Q":0,
    "Embarked_S":0
}

# Function for preprocessing data
def preprocess_data(df, input_type):

    df["Dependants"] = df["SibSp"]+df["Parch"]
    
    # Fill missing values with the mean from default_means
    for col in df.columns:
        df[col] = df[col].fillna(default_means[col])
        
    
    # One-hot encoding
    # One-hot encoding for categorical columns
    if input_type == "Upload CSV":
        df["Sex"]=df["Sex"].astype("category")
        df["Embarked"]= df["Embarked"].astype("category")
    
        categorical_cols = ["Sex", "Embarked"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

        if "Sex_female" in df.columns:
            df.drop(columns=["Sex_female"], inplace=True)
        if "Embarked_Unknown" in df.columns:
            df.drop(columns=["Embarked_Unknown"], inplace=True)
        
    # Scaling numeric columns
    numeric_cols = ["Age", "Fare"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

# Function for making predictions
def predict(data):
    prob = model.predict_proba(data)[:, 1]  # Get probability of class 1
    prediction = (prob >= 0.30).astype(int)  # Apply threshold
    return prediction, prob

# Streamlit UI
st.title("Logistic Regression Prediction App")
st.subheader("Assumption: Aiming for high Recall")
option = st.radio("Choose input method", ("Manual Entry", "Upload CSV"))

if option == "Manual Entry":
    st.subheader("Enter Features Manually")
    st.text("If the values are unknown keep the values in the boxes as it is, as it is the default values")
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0)
    fare = st.number_input("Fare", min_value=0.0, value=32.2)
    pclass = st.selectbox("Pclass", [1, 2, 3], index=1)
    sibsp = st.number_input("SibSp", min_value=0, value=0)
    parch = st.number_input("Parch", min_value=0, value=0)
    dependants = 0
    sex_male = st.selectbox("Sex", ["Male", "Female"]) == "Male"
    embarked = st.selectbox("Embarked", ["C", "Q", "S", "Unknown"])

    embarked_c = 1 if embarked == "C" else 0
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0
    
    # Create dataframe for processing
    input_data = pd.DataFrame([[age, fare, pclass, sibsp, parch, dependants, sex_male, embarked_c, embarked_q, embarked_s]],
                              columns=["Age", "Fare", "Pclass", "SibSp", "Parch", "Dependants", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"])
    
    processed_data = preprocess_data(input_data, option)
    
    if st.button("Predict"):
        pred, prob = predict(processed_data)
        st.write(f"Prediction: {pred[0]} (Probability: {prob[0]:.2f})")

elif option == "Upload CSV":
    st.subheader("Upload a CSV File")
    file = st.file_uploader("Upload CSV", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        df= df[["Age","Fare","Pclass","SibSp","Parch","Sex","Embarked"]]
        processed_df = preprocess_data(df, option)
        
        if st.button("Predict"):
            pred, prob = predict(processed_df)
            result_df = df.copy()
            result_df["Prediction"] = pred
            result_df["Probability"] = prob
            st.write(result_df)
            st.download_button("Download Predictions", result_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
