import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import os

# File paths
MODEL_PATH = "dtms_model.pkl"
TEST_DATA_PATH = "testwise_report_2025.xlsx"
DETAIL_DATA_PATH = "detail_report_2025.xlsx"

# Feature configuration
FEATURE_COLS = ['test_title', 'manufacturer', 'drug_form', 'sample_mode', 'days_in_dtl']
CATEGORICAL_COLS = ['test_title', 'manufacturer', 'drug_form', 'sample_mode']


# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=3600)
def load_data():
    """Load and merge testwise and detail reports from Excel files."""
    
    if not os.path.exists(TEST_DATA_PATH) or not os.path.exists(DETAIL_DATA_PATH):
        st.error("Dataset files not found in workspace directory.")
        return pd.DataFrame()

    df_test = pd.read_excel(TEST_DATA_PATH)
    df_detail = pd.read_excel(DETAIL_DATA_PATH)

    # Merge datasets
    df = pd.merge(df_test, df_detail, on='form6_barcode', how='left').drop_duplicates()

    # Fix duplicate columns
    if 'generic_name_x' in df.columns:
        df['generic_name'] = df['generic_name_x'].fillna(df.get('generic_name_y', 'Unknown'))

    # Convert date columns
    date_cols = [
        'received_datetime', 'issued_datetime', 'reported_datetime',
        'parcel_date', 'form6_date', 'form7_date'
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calculate TAT
    if 'tat_days' not in df.columns or df['tat_days'].isnull().all():
        if 'reported_datetime' in df.columns and 'received_datetime' in df.columns:
            df['tat_days'] = (df['reported_datetime'] - df['received_datetime']).dt.days
        elif 'turnaround_days' in df.columns:
            df['tat_days'] = df['turnaround_days']

    # Handle days_in_dtl
    if 'days_in_dtl' not in df.columns:
        df['days_in_dtl'] = df['tat_days'].fillna(0)
    else:
        df['days_in_dtl'] = pd.to_numeric(df['days_in_dtl'], errors='coerce').fillna(0)

    # Target column
    if 'is_delayed' not in df.columns:
        df['is_delayed'] = (df['tat_days'] > 30).astype(int)

    # Clean categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        else:
            df[col] = 'Unknown'

    return df


# =========================
# LOAD MODEL + ENCODERS
# =========================
@st.cache_resource
def load_model_and_encoders(df):
    """Load model and create encoders."""
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        return None, {}

    model = joblib.load(MODEL_PATH)

    # Create encoders
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        unique_vals = list(set(df[col].dropna().unique().tolist() + ['Unknown', 'Other']))
        le.fit(unique_vals)
        encoders[col] = le

    return model, encoders


# =========================
# ENCODING FUNCTION
# =========================
def encode_sample(sample_df, encoders):
    """Encode categorical features safely."""
    
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

    encoded_df['days_in_dtl'] = pd.to_numeric(
        encoded_df['days_in_dtl'], errors='coerce'
    ).fillna(0)

    return encoded_df[FEATURE_COLS]


# =========================
# SINGLE PREDICTION
# =========================
def predict_single(model, encoders, sample_dict):
    """Predict for one sample."""
    
    sample_df = pd.DataFrame([sample_dict])
    X = encode_sample(sample_df, encoders)

    pred_class = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    delay_prob = probabilities[1] if len(probabilities) > 1 else float(pred_class)

    return {
        'is_delayed': int(pred_class),
        'delay_probability': float(delay_prob),
        'risk_label': (
            'High Risk' if delay_prob >= 0.5
            else 'Moderate Risk' if delay_prob >= 0.3
            else 'Low Risk'
        )
    }


# =========================
# BATCH PREDICTION
# =========================
def predict_batch_data(model, encoders, batch_df):
    """Predict for multiple records."""
    
    sample_df = batch_df.copy()

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

    result_df = batch_df.copy()
    result_df['Predicted_Delayed'] = preds
    result_df['Delay_Probability_%'] = (probs * 100).round(2)

    result_df['Risk_Level'] = np.where(
        probs >= 0.5, 'High Risk',
        np.where(probs >= 0.3, 'Moderate Risk', 'Low Risk')
    )

    return result_df
