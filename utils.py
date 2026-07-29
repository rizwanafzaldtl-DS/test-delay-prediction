import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "dtms_model.pkl"
TEST_DATA_PATH = "testwise_report_2025.xlsx"
DETAIL_DATA_PATH = "detail_report_2025.xlsx"

FEATURE_COLS = ['test_title', 'manufacturer', 'drug_form', 'sample_mode', 'days_in_dtl']
CATEGORICAL_COLS = ['test_title', 'manufacturer', 'drug_form', 'sample_mode']

@st.cache_data(ttl=3600)
def load_data():
    """Load and merge testwise and detail reports from Excel files."""
    if not os.path.exists(TEST_DATA_PATH) or not os.path.exists(DETAIL_DATA_PATH):
        st.error("Dataset files not found in workspace directory.")
        return pd.DataFrame()
    
    df_test = pd.read_excel("testwise_report_2025.xlsx")
    df_detail = pd.read_excel("detail_report_2025.xlsx")
    
    # Merge datasets
    df = pd.merge(df_test, df_detail, on='form6_barcode', how='left').drop_duplicates()
    
    # Clean column names (handle duplicate merge columns if any)
    if 'generic_name_x' in df.columns:
        df['generic_name'] = df['generic_name_x'].fillna(df.get('generic_name_y', 'Unknown'))
    
    # Datetime conversions
    date_cols = ['received_datetime', 'issued_datetime', 'reported_datetime', 'parcel_date', 'form6_date', 'form7_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    # Calculate TAT in days if not present
    if 'tat_days' not in df.columns or df['tat_days'].isnull().all():
        if 'reported_datetime' in df.columns and 'received_datetime' in df.columns:
            df['tat_days'] = (df['reported_datetime'] - df['received_datetime']).dt.days
        elif 'turnaround_days' in df.columns:
            df['tat_days'] = df['turnaround_days']
            
    # Fill missing days_in_dtl if needed
    if 'days_in_dtl' not in df.columns:
        df['days_in_dtl'] = df['tat_days'].fillna(0)
    else:
        df['days_in_dtl'] = pd.to_numeric(df['days_in_dtl'], errors='coerce').fillna(0)
        
    # Target column is_delayed (TAT > 30 days)
    if 'is_delayed' not in df.columns:
        df['is_delayed'] = (df['tat_days'] > 30).astype(int)
        
    # Clean text columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        else:
            df[col] = 'Unknown'
            
    return df

@st.cache_resource
def load_model_and_encoders(df):
    """Load model and build LabelEncoders fitted on dataset categories."""
    model = joblib.load("dtms_model.pkl")
encoders = {}
    
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on unique dataset values plus 'Unknown'
        unique_vals = list(set(df[col].dropna().unique().tolist() + ['Unknown', 'Other']))
        le.fit(unique_vals)
        encoders[col] = le
        
    return model, encoders

def encode_sample(sample_df, encoders):
    """Encode categorical features in sample dataframe handling unknown values."""
    encoded_df = sample_df.copy()
    for col in CATEGORICAL_COLS:
        if col in encoded_df.columns:
            le = encoders[col]
            known_classes = set(le.classes_)
            encoded_df[col] = encoded_df[col].apply(
                lambda val: val if str(val) in known_classes else 'Unknown'
            )
            encoded_df[col] = le.transform(encoded_df[col].astype(str))
        else:
            encoded_df[col] = encoders[col].transform(['Unknown'] * len(encoded_df))
            
    encoded_df['days_in_dtl'] = pd.to_numeric(encoded_df['days_in_dtl'], errors='coerce').fillna(0)
    return encoded_df[FEATURE_COLS]

def predict_single(model, encoders, sample_dict):
    """Predict delay status and probability for a single sample dict."""
    sample_df = pd.DataFrame([sample_dict])
    X = encode_sample(sample_df, encoders)
    
    pred_class = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    delay_prob = probabilities[1] if len(probabilities) > 1 else float(pred_class)
    
    return {
        'is_delayed': int(pred_class),
        'delay_probability': float(delay_prob),
        'risk_label': 'High Risk' if delay_prob >= 0.5 else ('Moderate Risk' if delay_prob >= 0.3 else 'Low Risk')
    }

def predict_batch_data(model, encoders, batch_df):
    """Predict delay status and probability for a batch dataframe."""
    sample_df = batch_df.copy()
    
    # Ensure required columns exist
    for col in CATEGORICAL_COLS:
        if col not in sample_df.columns:
            sample_df[col] = 'Unknown'
        else:
            sample_df[col] = sample_df[col].fillna('Unknown').astype(str)
            
    if 'days_in_dtl' not in sample_df.columns:
        sample_df['days_in_dtl'] = 0
        
    X = encode_sample(sample_df, encoders)
    
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1] > 1 else preds
    
    res_df = batch_df.copy()
    res_df['Predicted_Delayed'] = preds
    res_df['Delay_Probability_%'] = (probs * 100).round(2)
    res_df['Risk_Level'] = np.where(probs >= 0.5, 'High Risk', np.where(probs >= 0.3, 'Moderate Risk', 'Low Risk'))
    
    return res_df
