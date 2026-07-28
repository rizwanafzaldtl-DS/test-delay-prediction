# 🧪 Drug Testing Management System (DTMS) - Machine Learning & Analytics WebApp

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

A comprehensive Machine Learning platform and interactive web application designed for the **Drug Testing Management System (DTMS)**. The system analyzes drug testing turnaround times (TAT), identifies operational bottlenecks, and predicts sample testing delay risks to accelerate quality assurance reporting and SLA compliance.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Machine Learning Model Performance](#-machine-learning-model-performance)
- [Dataset Overview](#-dataset-overview)
- [Installation & Local Setup](#-installation--local-setup)
- [Directory Structure](#-directory-structure)

---

## 🎯 Executive Overview

Drug Testing Laboratories (DTLs) process thousands of pharmaceutical samples annually across various drug forms (tablets, syrups, injectables, bandages) and testing requirements. Delayed testing reports impact drug supply chains, regulatory oversight, and public safety.

This project integrates data science and machine learning to:
1. **Automate Data Integration**: Merge multi-source testing logs (`testwise_report_2025` and `detail_report_2025`).
2. **Predict Testing Delays**: Utilize a trained Random Forest model (**93.11% accuracy**) to evaluate sample delay risk (>30-day turnaround time).
3. **Provide Proactive Decision Support**: Flag high-risk samples upon arrival to enable priority laboratory queueing.
4. **Offer Interactive Management Dashboards**: Enable lab directors to dynamically re-evaluate SLA compliance targets (7 to 60 days).

---

## ✨ Key Features

### 📊 1. Executive Analytics Dashboard
- **KPI Metrics**: Real-time evaluation of Total Samples (29,914), SLA Breach Count, Delay Rate %, and Average Turnaround Time (13.6 days).
- **Interactive SLA Slider**: Dynamically adjust SLA targets (7 to 60 days) to recalculate compliance metrics in real-time.
- **Visual Analytics**: Interactive Plotly charts for TAT distribution histograms, delay proportion pie charts, top delayed manufacturers, and sample mode breakdowns.

### 🔄 2. Project & Workflow Architecture
- Visual step-by-step breakdown of the 5-stage data & ML pipeline and 4-stage operational decision flow.

### 🔮 3. Single Sample Delay Predictor
- Select drug sample parameters (*Test Title, Manufacturer, Drug Form, Sample Mode, Days in DTL*) to calculate real-time delay probability score % and color-coded risk alerts.

### 📁 4. Batch Processing & Data Export
- Drag-and-drop file uploader for Excel (`.xlsx`) or CSV (`.csv`) batch files. Generates automated delay predictions and downloadable CSV reports.

### 📈 5. Model Insights & Feature Importance
- Visual feature importance weights and classification evaluation metrics.

### 🔍 6. Data Explorer
- Searchable and filterable dataset view with custom column selection.

---

## 🔄 Project Architecture & Workflow

```
[ testwise_report_2025.xlsx ] ──┐
                                ├──► [ Merge via form6_barcode ] ──► [ Clean & Preprocess ]
[ detail_report_2025.xlsx   ] ──┘                                        │
                                                                         ▼
                                                            [ Feature Engineering: tat_days ]
                                                                         │
                                                                         ▼
                                                            [ Target: is_delayed (TAT > 30d) ]
                                                                         │
                                                                         ▼
                                                            [ LabelEncoding Categoricals ]
                                                                         │
                                                                         ▼
                                                            [ Random Forest Model (150 trees) ]
                                                                         │
                                                                         ▼
                                                            [ Streamlit Web App (app.py) ]
```

---

## 📈 Machine Learning Model Performance

- **Algorithm**: Random Forest Classifier (`n_estimators=150`)
- **Validation Accuracy**: **93.11%** (on unseen 20% test split)
- **Primary Feature Importance Drivers**:
  1. **`days_in_dtl` (32.8%)**: Laboratory elapsed processing days.
  2. **`manufacturer` (30.4%)**: Manufacturing company historical risk.
  3. **`sample_mode` (17.6%)**: Mode of sample collection/delivery.
  4. **`drug_form` (11.8%)**: Dosage form complexity.
  5. **`test_title` (7.4%)**: Chemical/microbiological assay type.

---

## 📁 Dataset Overview

The system processes two primary data sources joined by `form6_barcode`:
- `testwise_report_2025.xlsx`: Contains sample barcodes, test titles, received/issued/reported timestamps, lab codes, and turnaround days.
- `detail_report_2025.xlsx`: Contains sample collection metadata, parcel dates, drug forms, batch numbers, manufacturers, and sample modes.

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Clone / Navigate to Directory
```bash
cd "d:/Data Science & AI/PowerBI projects/DTL/DTMS antigravity"
```

### 3. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 4. Launch Streamlit Application
```bash
py -m streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 📂 Directory Structure

```
DTMS antigravity/
├── app.py                      # Main Streamlit Web Application
├── utils.py                    # Helper pipeline (data loading, encoding, prediction)
├── dtms_model.pkl              # Trained Random Forest ML model
├── testwise_report_2025.xlsx   # Testwise laboratory dataset
├── detail_report_2025.xlsx     # Sample details dataset
├── DTMS_project.ipynb          # Model training notebook
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```
