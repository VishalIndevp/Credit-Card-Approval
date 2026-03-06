# 💳 Credit Card Approval Prediction
### Artificial Neural Network · TensorFlow/Keras · Streamlit

> Predict credit card approval likelihood in real-time using a trained ANN model — deployable as an interactive web app.

---

## 🧠 Overview

This project builds an **Artificial Neural Network (ANN)** to classify whether a credit card application will be **approved or rejected**, based on applicant profile data such as income, employment history, family status, and housing type.

A **Streamlit web app** lets users enter applicant details and receive instant predictions — making the model accessible without any coding knowledge.

---

## 📁 Project Structure

```
Credit-Card-Approval-ANN/
│
├── app.py                  # Streamlit web application
├── application_record.csv  # Raw dataset
├── ann_model.h5            # Trained ANN model
├── X_min.pkl               # Min values for normalization
├── X_max.pkl               # Max values for normalization
├── columns.pkl             # Feature column names
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Dataset

**File:** `application_record.csv`

### Features Used

| Feature | Description |
|---|---|
| `CODE_GENDER` | Applicant gender |
| `FLAG_OWN_CAR` | Car ownership |
| `FLAG_OWN_REALTY` | Property ownership |
| `CNT_CHILDREN` | Number of children |
| `AMT_INCOME_TOTAL` | Annual income |
| `NAME_INCOME_TYPE` | Income source type |
| `NAME_EDUCATION_TYPE` | Education level |
| `NAME_FAMILY_STATUS` | Marital status |
| `NAME_HOUSING_TYPE` | Housing situation |
| `DAYS_BIRTH` → `AGE` | Applicant age (converted) |
| `DAYS_EMPLOYED` → `EMPLOYMENT_YEARS` | Years employed (converted) |
| `FLAG_MOBIL` | Mobile phone flag |
| `FLAG_WORK_PHONE` | Work phone flag |
| `FLAG_PHONE` | Phone flag |
| `FLAG_EMAIL` | Email flag |
| `OCCUPATION_TYPE` | Type of occupation |
| `CNT_FAM_MEMBERS` | Family member count |

### Target Variable

| Value | Meaning |
|---|---|
| `1` | ✅ Credit card approved |
| `0` | ❌ Credit card not approved |

---

## ⚙️ Data Preprocessing

### 1. One-Hot Encoding
Categorical variables were encoded into numeric form:
```python
pd.get_dummies()
```

### 2. Feature Engineering
Two date-based columns were transformed into interpretable features:
```python
AGE             = DAYS_BIRTH    → converted to years
EMPLOYMENT_YEARS = DAYS_EMPLOYED → converted to years
```

### 3. Min-Max Normalization
Applied manually to scale all features to [0, 1]:
```python
X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
```

### 4. Train/Test Split
```python
train_test_split(test_size=0.33, stratify=y)
```

---

## 🏗️ Model Architecture

```
Input Layer
    ↓
Dense(32, activation='relu')
    ↓
Dense(16, activation='relu')
    ↓
Dense(1, activation='sigmoid')  →  Output: Approval Probability
```

| Parameter | Value |
|---|---|
| Loss Function | `binary_crossentropy` |
| Optimizer | `SGD` |
| Epochs | 100 |
| Batch Size | 1000 |

---

## 📈 Model Performance

Final epoch results (Epoch 100/100):

```
294/294 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step
accuracy:     0.9817     loss:     0.0574
val_accuracy: 0.9818     val_loss: 0.0570
```

| Metric | Training | Validation |
|---|---|---|
| Accuracy | **98.17%** | **98.18%** |
| Loss | 0.0574 | 0.0570 |

> ✅ The model generalizes exceptionally well — near-identical training and validation accuracy indicates no overfitting.

---

## 🌐 Streamlit Web Application

The app provides a clean UI to:
1. Enter applicant details via input fields
2. Preprocess inputs using the saved pipeline
3. Predict approval in real-time

**Sample outputs:**
```
✅ Loan Approved
❌ Loan Not Approved
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/VishalIndevp/credit-card-approval-ann.git
cd credit-card-approval-ann
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | ANN model building & training |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-Learn | Train/test split |
| Streamlit | Web app interface |
| Joblib | Model & pipeline serialization |

---

## 👤 Author

**Vishal Singh**  
*Data Analytics & AI Enthusiast*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/vishal-singh-here/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/VishalIndevp)

---

## 📝 Note

> This project is built for **learning and portfolio purposes**, demonstrating how a machine learning model can be trained, saved, and deployed as an interactive web application using Streamlit.
