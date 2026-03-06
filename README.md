# Credit Card Approval Prediction using Artificial Neural Network (ANN)

## Overview

This project predicts whether a **credit card application will be approved or not** using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.

The model analyzes applicant information such as **income, employment, family status, housing type, and occupation** to determine approval likelihood.

A **Streamlit web application** is included so users can enter applicant details and get real-time predictions.

---

## Dataset

The dataset used is **application_record.csv**.

### Features used in the model

* CODE_GENDER
* FLAG_OWN_CAR
* FLAG_OWN_REALTY
* CNT_CHILDREN
* AMT_INCOME_TOTAL
* NAME_INCOME_TYPE
* NAME_EDUCATION_TYPE
* NAME_FAMILY_STATUS
* NAME_HOUSING_TYPE
* DAYS_BIRTH → converted to **AGE**
* DAYS_EMPLOYED → converted to **EMPLOYMENT_YEARS**
* FLAG_MOBIL
* FLAG_WORK_PHONE
* FLAG_PHONE
* FLAG_EMAIL
* OCCUPATION_TYPE
* CNT_FAM_MEMBERS

### Target Column

`approved`

* **1 → Credit card approved**
* **0 → Credit card not approved**

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. One-Hot Encoding

Categorical variables were converted into numerical form using:

```
pd.get_dummies()
```

### 2. Feature Engineering

Two columns were transformed:

* `DAYS_BIRTH` → **AGE**
* `DAYS_EMPLOYED` → **EMPLOYMENT_YEARS**

### 3. Normalization

Manual **Min-Max normalization** was applied:

```
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
```

### 4. Train Test Split

```
train_test_split(test_size = 0.33, stratify = y)
```

---

## Model Architecture

The model is an **Artificial Neural Network (ANN)**.

Architecture:

* Input Layer
* Dense Layer (32 neurons, ReLU)
* Dense Layer (16 neurons, ReLU)
* Output Layer (1 neuron, Sigmoid)

Loss Function:

```
binary_crossentropy
```

Optimizer:

```
SGD
```

Training:

* **Epochs:** 100
* **Batch Size:** 1000

---

## Model Performance

Example training result:

```
accuracy: 0.73
loss: 0.49
validation accuracy: ~0.59
```

---

## Streamlit Web Application

A Streamlit interface allows users to:

1. Enter applicant information
2. Process input using the same preprocessing pipeline
3. Predict credit approval instantly

Example outputs:

```
✅ Loan Approved
❌ Loan Not Approved
```

---

## Project Structure

```
Credit-Card-Approval-ANN
│
├── app.py
├── application_record.csv
├── ann_model.h5
├── X_min.pkl
├── X_max.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/credit-card-approval-ann.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit app:

```
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

---

## Author

**Vishal Singh**

Data Analytics & AI Enthusiast

LinkedIn:
https://www.linkedin.com/in/vishal-singh-here/

GitHub:
https://github.com/VishalIndevp

---

## Note

This project is built for **learning and portfolio purposes**, demonstrating how machine learning models can be deployed with an interactive web interface using Streamlit.
