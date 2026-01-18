#  Heart Disease Prediction System

An end-to-end Machine Learning application that predicts the likelihood of heart disease based on patient clinical data.  
The project includes data analysis, model training, evaluation, and deployment using Streamlit.

---

##  Problem Statement
Heart disease is one of the leading causes of death worldwide. Early detection can significantly improve treatment outcomes.  
This project aims to build a machine learning model that predicts whether a patient is likely to have heart disease based on clinical parameters.

---

##  Dataset
- **Source:** UCI Heart Disease Dataset (Kaggle)
- **Type:** Structured medical dataset
- **Target Variable:**  
  - `0` → No heart disease  
  - `1` → Heart disease present

### Features include:
- Age, Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Maximum heart rate
- Exercise-induced angina
- ST depression, slope
- Number of major vessels
- Thalassemia

---

##  Exploratory Data Analysis (EDA)
Key EDA steps performed:
- Checked class balance (dataset is fairly balanced)
- Correlation analysis using heatmap
- Feature vs target analysis using boxplots
- Identified key predictive features such as chest pain type, exercise-induced angina, ST depression, and max heart rate

---

##  Model Building
Two models were trained and compared:

### 1️ Logistic Regression (Final Model ✅)
- Accuracy: ~85%
- Balanced precision and recall
- Fewer false negatives (important in medical use cases)
- Better generalization on small, structured datasets

### 2️ Random Forest
- Lower accuracy (~70%)
- High false negatives
- Poor recall for heart disease class

 **Logistic Regression was selected** due to better performance and medical relevance.

---

##  Model Evaluation
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score

Special emphasis was placed on **minimizing false negatives**, as missing a heart disease case is critical.

---

##  Deployment
The trained model was deployed using **Streamlit** to create an interactive web application where users can:
- Enter patient clinical data
- Get real-time prediction
- View prediction confidence (probability)

---

##  Project Structure
heart-disease-prediction/
├── app.py
├── logistic_model.pkl
├── scaler.pkl
├── notebooks/
│ ├── EDA.ipynb
│ └── model_training.ipynb
├── data/
│ └── heart.csv
├── requirements.txt
└── README.md


---

##  How to Run the App Locally

### 1️ Clone the repository
```bash
git clone (https://github.com/bhatiaanmol/Heart_Disease_Prediction)
cd Heart_Disease_Prediction

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt

python -m streamlit run app.py
