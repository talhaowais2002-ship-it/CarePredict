from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Pandas will be utilized for data manipulation as specified

# Dataset is loaded, here it will be the UCI Diabetes dataset
# Dataset is chosen as diabetes is a multi-system disease affecting heart, kidneys, nerves etc.
# Hence a trained model for diabetes can be used as the blueprint for other diseases going forward


def load_and_initialize(file_path):
    # df is the variable name for Pandas library structure
    df = pd.read_csv(file_path)
    # Initial exploration to understand 'Black Box' aspect of data
    print(f"Datset Shape: {df.shape}")
    return df

# Step 1.1 = Handle Missing Values
# Research highlights that healthcare data can often have missing or repeated entires, shown as '?' here


def handle_missing_data(df):
    # nan, Not A Number is a special floating value used to showcase undefined or missing numerical data
    df.replace('?', np.nan, inplace=True)
    # These are examples of pointless data as mentioned in UCI Diabetes dataset
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty']
    df_cleaned = df.drop(columns=cols_to_drop)
    # Dropping rows with null values
    df_cleaned.dropna(subset=['diag_1'], inplace=True)
    df_cleaned.fillna('Unknown', inplace=True)
    return df_cleaned


def feature_engineering(df):
    # 1. Target Variable Definition
    # Research focuses on 30 day readmission, we convert <30 into 1 (Risk) and others to 0 (No Risk)
    # Thus we get a binary classification
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    # 2. Age Transformation
    # Dataset provides age range in brackets such as 10-20 etc.
    # Midpoints are converted to make the numerical for algorithms
    age_dict = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
                '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
    if 'age' in df.columns:
        df['age'] = df['age'].replace(age_dict)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    print('Age column converted to numeric successfully')
    # 3. Categorical Encoding
    # As mentioned, categorical variables need to be encoded, here we use label encoding for binary categories
    le = LabelEncoder()
    categorical_cols = ['race', 'gender',
                        'change', 'diabetesMed']  # change is 'change in medication'
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df


def prepare_for_training(df):
    df = df.select_dtypes(include=[np.number])
    # 1. Seperating Features(X) and Target(Y)
    # Drop target column and 'encounter_id'/'patient_nbr'as they are just IDs dont help in predictions
    X = df.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])
    Y = df['readmitted']
    # 2. Train-Test Split
    # We use a 80/20 split as standard in academic research
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y)
    # 3. Feature Scaling
    # Logistic Regression is sensitive to the scale of data (such as age vs number)
    # StandardScaler ensures all features have a mean of 0 and variance of 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, Y_train, Y_test, X.columns, scaler


def train_models(X_train, Y_train):
    custom_weights = {0: 1, 1: 8}
    # 1.Logistic Regression
    # max_iter=1000 gives the solver more time to find sweet spot
    # class_weight='balanced' prevents model from ignoring minority 'at-risk' patients
    lr_model = LogisticRegression(
        max_iter=1000, class_weight=custom_weights, random_state=42)
    lr_model.fit(X_train, Y_train)
    # 2.Random Forest
    # n_estimaters=100 creates 100 Decision Trees to vote on the readmission outcome
    # This is the "Ensemble Method"
    rf_model = RandomForestClassifier(
        n_estimators=100, class_weight=custom_weights, max_depth=10, random_state=42)
    rf_model.fit(X_train, Y_train)
    return lr_model, rf_model


def evaluate_performance(model, X_test, Y_test, model_name):
    # Use trained model to predict outcomes for 20% unseen test data
    predictions = model.predict(X_test)
    print(f"--- {model_name} Performance ---")
    # classification_report provides the Precision, Recall and F1-Score
    print(classification_report(Y_test, predictions))
    # confusion_matrix shows how many True Positives VS False Negatives are obtained
    return confusion_matrix(Y_test, predictions)


def visualize_results(model, X_columns):
    # 1. Feature Importance Plot for addressing the 'Black Box' problem
    # This shows which medical factors actually drive readmission risk
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 factors
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Clinical Factors Affecting Readmission Risk')
    plt.barh(range(len(indices)),
             importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [X_columns[i] for i in indices])
    plt.xlabel('Relative Importance Score')
    plt.tight_layout()
    plt.show()

# ========================================
# FINAL EXECUTION: PIPELINE STARTING POINT
# ========================================


# 1. Load Dataset
data = load_and_initialize('diabetic_data.csv')
# 2.Clean Data (removing '?' and uneeded columns)
data_cleaned = handle_missing_data(data)
# 3. Transform Data (Age Mapping and Label Encoding)
data_ready = feature_engineering(data_cleaned)
# 4. Prepare for AI (Splaitting into 80/20 and Scaling)
X_train, X_test, Y_train, Y_test, feature_names, hospital_scaler = prepare_for_training(
    data_ready)
# 5. Train both models for Comparative Analysis
# Note: Core Experiment of Research
log_model, forest_model = train_models(X_train, Y_train)
# 6. Evaliate and Print Results to Terminal
evaluate_performance(log_model, X_test, Y_test, 'Logistic Regression')
evaluate_performance(forest_model, X_test, Y_test, 'Random Forest')
# 7. Generate Visual Chart for 'Black Box' reasoning
visualize_results(forest_model, feature_names)

joblib.dump(log_model, 'carepredict_model.pkl')
joblib.dump(hospital_scaler, 'hospital_scaler.pkl')
joblib.dump(feature_names, 'feature_metadata.pkl')
print('System Assets Saved: carepredict_model.pkl & hospital_scaler.pkl')
#Run with 'python CarePredict.py' to train model