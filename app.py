import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Load model and preprocessing files
model = keras.models.load_model("ann_model.h5")
X_min = joblib.load("X_min.pkl")
X_max = joblib.load("X_max.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Loan Approval Prediction")
st.write("Enter applicant details to predict approval.")

# ---------- USER INPUT ----------

gender = st.selectbox("Gender", ["M","F"])
car = st.selectbox("Own Car", ["Y","N"])
realty = st.selectbox("Own Realty", ["Y","N"])

children = st.number_input("Number of Children",0,10,0)

income = st.number_input("Total Income",10000,1000000,50000)

income_type = st.selectbox(
    "Income Type",
    ["Working","Commercial associate","Pensioner","State servant","Student"]
)

education = st.selectbox(
    "Education",
    ["Secondary / secondary special",
     "Higher education",
     "Incomplete higher",
     "Lower secondary",
     "Academic degree"]
)

family_status = st.selectbox(
    "Family Status",
    ["Single / not married","Married","Civil marriage","Separated","Widow"]
)

housing = st.selectbox(
    "Housing Type",
    ["House / apartment","With parents","Municipal apartment","Rented apartment","Office apartment"]
)

age = st.number_input("Age",18,100,30)

employment_years = st.number_input("Employment Years",0,50,5)

mobil = st.selectbox("Has Mobile", [0,1])
work_phone = st.selectbox("Work Phone", [0,1])
phone = st.selectbox("Phone", [0,1])
email = st.selectbox("Email", [0,1])

occupation = st.selectbox(
    "Occupation",
    ["Laborers","Core staff","Accountants","Managers","Drivers","Sales staff",
     "Cleaning staff","Cooking staff","Medicine staff","Private service staff"]
)

family_members = st.number_input("Family Members",1,10,2)

# ---------- CREATE INPUT DATA ----------

input_dict = {
    "CNT_CHILDREN":[children],
    "AMT_INCOME_TOTAL":[income],
    "AGE":[age],
    "EMPLOYMENT_YEARS":[employment_years],
    "FLAG_MOBIL":[mobil],
    "FLAG_WORK_PHONE":[work_phone],
    "FLAG_PHONE":[phone],
    "FLAG_EMAIL":[email],
    "CNT_FAM_MEMBERS":[family_members],
    
    "CODE_GENDER":[gender],
    "FLAG_OWN_CAR":[car],
    "FLAG_OWN_REALTY":[realty],
    "NAME_INCOME_TYPE":[income_type],
    "NAME_EDUCATION_TYPE":[education],
    "NAME_FAMILY_STATUS":[family_status],
    "NAME_HOUSING_TYPE":[housing],
    "OCCUPATION_TYPE":[occupation]
}

input_df = pd.DataFrame(input_dict)

# Apply one hot encoding
input_df = pd.get_dummies(input_df)

# Align with training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Normalization
input_scaled = (input_df - X_min) / (X_max - X_min + 1e-8)

# ---------- PREDICTION ----------

if st.button("Predict"):

    prediction = model.predict(input_scaled.values)[0][0]

    if prediction > 0.5:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

st.write("Prediction Probability:", prediction if 'prediction' in locals() else "")