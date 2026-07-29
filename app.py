import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import utils
import os
import pandas as pd
import os
import streamlit as st
import gdown
import os
import joblib
# Custom CSS for Modern Premium Aesthetic
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Card Component */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Workflow Step Cards */
    .workflow-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .workflow-step-num {
        font-size: 0.8rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .workflow-step-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0 6px 0;
    }
    .workflow-step-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Prediction Container */
    .pred-box-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .pred-box-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(4, 120, 87, 0.25));
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    /* Header title styling */
    .app-header {
        background: linear-gradient(90deg, #0ea5e9, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
df_raw = utils.load_data()

if df_raw.empty:
    st.error("Failed to load DTMS datasets. Please check file paths.")
    st.stop()

model, encoders = utils.load_model_and_encoders(df_raw)

# Sidebar Layout
st.sidebar.image("https://img.icons8.com/isometric-line/100/laboratory.png", width=70)
st.sidebar.markdown("## 🧪 DTMS Analytics")
st.sidebar.caption("Drug Testing Management System v2.0")
st.sidebar.markdown("---")

# Global Dataset Filters in Sidebar
st.sidebar.subheader("📌 Dashboard Filters")

# SLA Threshold Slider (Interactive Feature)
sla_days = st.sidebar.slider(
    "Target SLA Threshold (Days)",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
    help="Adjust target SLA to dynamically re-evaluate delay percentages across the dataset."
)

# Filter by Sample Mode if available
available_modes = ["All"] + sorted(list(df_raw['sample_mode'].unique()))
selected_mode = st.sidebar.selectbox("Filter by Sample Mode", available_modes)

# Filter by Drug Form
available_forms = ["All"] + sorted(list(df_raw['drug_form'].unique()))
selected_form = st.sidebar.selectbox("Filter by Drug Form", available_forms)

# Apply filters
df = df_raw.copy()
if selected_mode != "All":
    df = df[df['sample_mode'] == selected_mode]
if selected_form != "All":
    df = df[df['drug_form'] == selected_form]

# Dynamic delay calculation based on SLA slider
df['dynamic_delayed'] = (df['tat_days'] > sla_days).astype(int)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About Project")
st.sidebar.info("""
**DTMS Machine Learning Project**
Predicts drug testing sample turnaround delays to optimize lab efficiency, identify bottlenecks, and accelerate quality assurance reporting.
""")

# Main Header
st.markdown("<div class='app-header'>Drug Testing Management System (DTMS)</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Machine Learning Workflow Analysis, SLA Optimization & Delay Prediction Platform</div>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Dashboard",
    "🔄 Project & Workflow Architecture",
    "🔮 Single Test Predictor",
    "📁 Batch Delay Prediction",
    "📈 Model Performance",
    "🔍 Data Explorer"
])

# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tab1:
    st.markdown("### 📊 Workflow & Delay KPI Dashboard")
    
    # Key Performance Indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = len(df)
    delayed_samples = int(df['dynamic_delayed'].sum())
    delay_rate = (delayed_samples / total_samples * 100) if total_samples > 0 else 0
    avg_tat = df['tat_days'].mean() if 'tat_days' in df.columns else 0
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Samples Evaluated</div>
            <div class='metric-value'>{total_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Delayed Count (>{sla_days}d SLA)</div>
            <div class='metric-value' style='color:#ef4444;'>{delayed_samples:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>SLA Breach Rate</div>
            <div class='metric-value' style='color:#f59e0b;'>{delay_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Average Turnaround (TAT)</div>
            <div class='metric-value' style='color:#10b981;'>{avg_tat:.1f} Days</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row 1
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        st.markdown("#### Turnaround Time (TAT) Distribution")
        tat_clean = df['tat_days'].dropna()
        tat_plot = tat_clean[tat_clean <= 180]
        
        fig_tat = px.histogram(
            tat_plot,
            x='tat_days',
            nbins=40,
            color_discrete_sequence=['#38bdf8'],
            labels={'tat_days': 'Turnaround Time (Days)'},
            template='plotly_dark'
        )
        fig_tat.add_vline(x=sla_days, line_dash="dash", line_color="#ef4444", annotation_text=f"{sla_days}-Day SLA Line", annotation_position="top right")
        fig_tat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_tat, use_container_width=True)
        
    with v_col2:
        st.markdown(f"#### SLA Compliance (SLA Target: {sla_days} Days)")
        delay_counts = df['dynamic_delayed'].value_counts().reset_index()
        delay_counts.columns = ['Status', 'Count']
        delay_counts['Status'] = delay_counts['Status'].map({0: f'Compliant (≤{sla_days}d)', 1: f'SLA Breached (>{sla_days}d)'})
        
        fig_pie = px.pie(
            delay_counts,
            names='Status',
            values='Count',
            color='Status',
            color_discrete_map={f'Compliant (≤{sla_days}d)': '#10b981', f'SLA Breached (>{sla_days}d)': '#ef4444'},
            hole=0.4,
            template='plotly_dark'
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Visualizations Row 2
    st.markdown("<br>", unsafe_allow_html=True)
    v2_col1, v2_col2 = st.columns(2)
    
    with v2_col1:
        st.markdown("#### Top 10 Manufacturers by Delay Rate (Min 20 Samples)")
        mfg_stats = df.groupby('manufacturer').agg(
            total=('dynamic_delayed', 'count'),
            delayed=('dynamic_delayed', 'sum')
        ).reset_index()
        mfg_stats = mfg_stats[mfg_stats['total'] >= 20]
        mfg_stats['delay_rate'] = (mfg_stats['delayed'] / mfg_stats['total'] * 100).round(1)
        mfg_top = mfg_stats.sort_values(by='delay_rate', ascending=False).head(10)
        
        fig_mfg = px.bar(
            mfg_top,
            x='delay_rate',
            y='manufacturer',
            orientation='h',
            color='delay_rate',
            color_continuous_scale='Reds',
            labels={'delay_rate': 'Delay Rate (%)', 'manufacturer': 'Manufacturer'},
            template='plotly_dark'
        )
        fig_mfg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(autorange="reversed"),
            height=380
        )
        st.plotly_chart(fig_mfg, use_container_width=True)
        
    with v2_col2:
        st.markdown("#### Average Turnaround Time by Sample Mode")
        mode_stats = df.groupby('sample_mode').agg(
            avg_tat=('tat_days', 'mean'),
            count=('tat_days', 'count')
        ).reset_index().sort_values(by='avg_tat', ascending=False)
        
        fig_mode = px.bar(
            mode_stats,
            x='sample_mode',
            y='avg_tat',
            color='avg_tat',
            color_continuous_scale='Blues',
            labels={'avg_tat': 'Avg TAT (Days)', 'sample_mode': 'Sample Collection Mode'},
            template='plotly_dark'
        )
        fig_mode.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig_mode, use_container_width=True)

# ==========================================
# TAB 2: PROJECT & WORKFLOW ARCHITECTURE
# ==========================================
with tab2:
    st.markdown("### 🔄 DTMS Machine Learning Workflow Architecture")
    st.markdown("Comprehensive overview of the end-to-end data lifecycle, model training pipeline, and decision support workflow.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    w_col1, w_col2 = st.columns(2)
    
    with w_col1:
        st.markdown("#### 🛠️ Data & ML Pipeline Steps")
        
        st.markdown("""
        <div class='workflow-card'>
            <div class='workflow-step-num'>Step 01 • Data Integration</div>
            <div class='workflow-step-title'>Multi-Source Data Ingestion & Merging</div>
            <div class='workflow-step-desc'>Combines <code>testwise_report_2025.xlsx</code> (test parameters, dates, lab logs) and <code>detail_report_2025.xlsx</code> (sample specs, drug forms, manufacturers) via <code>form6_barcode</code>.</div>
        </div>
        
        <div class='workflow-card'>
            <div class='workflow-step-num'>Step 02 • Data Preprocessing & Cleaning</div>
            <div class='workflow-step-title'>Datetime Normalization & Deduplication</div>
            <div class='workflow-step-desc'>Standardizes date fields (received, issued, reported, parcel dates), removes duplicate entries, and handles missing text values gracefully.</div>
        </div>
        
        <div class='workflow-card'>
            <div class='workflow-step-num'>Step 03 • Feature Engineering</div>
            <div class='workflow-step-title'>Turnaround Time (TAT) & Delay Target</div>
            <div class='workflow-step-desc'>Computes turnaround duration <code>tat_days = reported_datetime - received_datetime</code> and sets the operational delay flag <code>is_delayed = 1</code> if TAT > 30 days.</div>
        </div>
        
        <div class='workflow-card'>
            <div class='workflow-step-num'>Step 04 • Feature Encoding</div>
            <div class='workflow-step-title'>Categorical Label Encoding</div>
            <div class='workflow-step-desc'>Applies <code>LabelEncoder</code> to convert high-cardinality text fields (<i>test_title, manufacturer, drug_form, sample_mode</i>) into numeric features for Random Forest modeling.</div>
        </div>
        
        <div class='workflow-card'>
            <div class='workflow-step-num'>Step 05 • Model Training & Evaluation</div>
            <div class='workflow-step-title'>Random Forest Classification</div>
            <div class='workflow-step-desc'>Trains a 150-tree Random Forest Classifier on 80% split data, achieving an accuracy score of <b>93.11%</b> on unseen test data.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with w_col2:
        st.markdown("#### 🎯 Operational Decision Support Flow")
        
        st.markdown("""
        <div class='workflow-card' style='border-left-color: #818cf8;'>
            <div class='workflow-step-num' style='color: #818cf8;'>Workflow Stage 1</div>
            <div class='workflow-step-title'>Sample Arrival & Entry Logging</div>
            <div class='workflow-step-desc'>Drug sample is registered at DTL laboratory. Metadata (drug form, batch, manufacturer, test type) is logged into DTMS.</div>
        </div>
        
        <div class='workflow-card' style='border-left-color: #818cf8;'>
            <div class='workflow-step-num' style='color: #818cf8;'>Workflow Stage 2</div>
            <div class='workflow-step-title'>Automated Delay Risk Scoring</div>
            <div class='workflow-step-desc'>The ML model calculates real-time delay probability score %. Samples with >50% probability are automatically flagged as High Delay Risk.</div>
        </div>
        
        <div class='workflow-card' style='border-left-color: #818cf8;'>
            <div class='workflow-step-num' style='color: #818cf8;'>Workflow Stage 3</div>
            <div class='workflow-step-title'>Priority Queue Assignment</div>
            <div class='workflow-step-desc'>High-risk samples receive expedited analyst allocation to prevent SLA breaches and minimize total turnaround time.</div>
        </div>
        
        <div class='workflow-card' style='border-left-color: #818cf8;'>
            <div class='workflow-step-num' style='color: #818cf8;'>Workflow Stage 4</div>
            <div class='workflow-step-title'>Continuous Monitoring & Reporting</div>
            <div class='workflow-step-desc'>Lab directors track operational bottlenecks, manufacturer performance, and turnaround metrics via interactive dashboards.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Project Goal Summary**: Transform static lab testing data into proactive decision support to ensure drug safety reports are delivered on schedule.")

# ==========================================
# TAB 3: SINGLE SAMPLE PREDICTOR
# ==========================================
with tab3:
    st.markdown("### 🔮 Single Drug Sample Delay Risk Evaluator")
    st.markdown("Select testing parameters below to evaluate whether a sample is at risk of exceeding the turnaround SLA threshold.")
    
    p_col1, p_col2 = st.columns([1.2, 1.0])
    
    with p_col1:
        st.markdown("#### 📝 Sample Details Form")
        
        test_titles = sorted(list(df_raw['test_title'].unique()))
        manufacturers = sorted(list(df_raw['manufacturer'].unique()))
        drug_forms = sorted(list(df_raw['drug_form'].unique()))
        sample_modes = sorted(list(df_raw['sample_mode'].unique()))
        
        sel_test = st.selectbox("Test Title", test_titles)
        sel_mfg = st.selectbox("Manufacturer", manufacturers)
        sel_form = st.selectbox("Drug Form", drug_forms)
        sel_mode = st.selectbox("Sample Mode", sample_modes)
        
        days_in_dtl_input = st.number_input(
            "Days in DTL (Laboratory Processing Days)",
            min_value=0,
            max_value=365,
            value=15,
            step=1,
            help="Estimated or current number of days sample has been processing in the lab."
        )
        
        predict_btn = st.button("🚀 Predict Delay Risk", type="primary", use_container_width=True)
        
    with p_col2:
        st.markdown("#### 🎯 Prediction Results")
        
        if predict_btn or 'last_prediction' in st.session_state:
            sample_dict = {
                'test_title': sel_test,
                'manufacturer': sel_mfg,
                'drug_form': sel_form,
                'sample_mode': sel_mode,
                'days_in_dtl': days_in_dtl_input
            }
            
            res = utils.predict_single(model, encoders, sample_dict)
            prob = res['delay_probability']
            prob_pct = prob * 100
            risk_label = res['risk_label']
            is_del = res['is_delayed']
            
            st.session_state['last_prediction'] = res
            
            # Risk Display Box
            if is_del == 1 or prob >= 0.5:
                st.markdown(f"""
                <div class='pred-box-high'>
                    <h2 style='color:#ef4444; margin:0;'>⚠️ HIGH RISK OF DELAY</h2>
                    <h1 style='color:#f8fafc; font-size:3.5rem; margin:10px 0;'>{prob_pct:.1f}%</h1>
                    <p style='color:#fca5a5; font-size:1.1rem;'>This drug test is predicted to exceed the 30-day turnaround threshold.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-box-low'>
                    <h2 style='color:#10b981; margin:0;'>✅ ON-TIME LIKELY</h2>
                    <h1 style='color:#f8fafc; font-size:3.5rem; margin:10px 0;'>{prob_pct:.1f}%</h1>
                    <p style='color:#6ee7b7; font-size:1.1rem;'>This drug test is expected to complete within normal limits (≤30 days).</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Delay Risk Probability (%)", 'font': {'size': 16, 'color': "#94a3b8"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#ef4444" if prob >= 0.5 else "#10b981"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 50], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"},
                height=250,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        else:
            st.info("👈 Fill in the parameters on the left and click **Predict Delay Risk** to see real-time model evaluation.")

# ==========================================
# TAB 4: BATCH DELAY PREDICTION
# ==========================================
with tab4:
    st.markdown("### 📁 Batch Sample Processing & CSV/Excel Upload")
    st.markdown("Upload a file containing multiple drug testing records to generate batch delay risk predictions.")
    
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV (.csv) file", type=['xlsx', 'csv'])
    
    st.markdown("##### Expected File Columns:")
    st.caption("`test_title`, `manufacturer`, `drug_form`, `sample_mode`, `days_in_dtl` (Additional columns will be preserved)")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)
                
            st.success(f"File uploaded successfully! Loaded {len(batch_df):,} records.")
            
            with st.spinner("Executing machine learning batch prediction..."):
                results_df = utils.predict_batch_data(model, encoders, batch_df)
                
            st.markdown("#### 📊 Prediction Results Summary")
            
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1:
                st.metric("Total Batch Records", f"{len(results_df):,}")
            with b_col2:
                high_risk_count = (results_df['Predicted_Delayed'] == 1).sum()
                st.metric("Predicted Delayed (>30d)", f"{high_risk_count:,}", delta=f"{high_risk_count/len(results_df)*100:.1f}%", delta_color="inverse")
            with b_col3:
                avg_batch_prob = results_df['Delay_Probability_%'].mean()
                st.metric("Avg Delay Risk Probability", f"{avg_batch_prob:.1f}%")
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)
            
            # Export options
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results (CSV)",
                data=csv_data,
                file_name="dtms_batch_predictions.csv",
                mime="text/csv",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

# ==========================================
# TAB 5: MODEL PERFORMANCE & METRICS
# ==========================================
with tab5:
    st.markdown("### 📈 Machine Learning Model Insights & Feature Importance")
    
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("#### 🏆 Random Forest Classifier Feature Importance")
        st.caption("Contribution of each feature in predicting sample turnaround delays.")
        
        feature_names = ['test_title', 'manufacturer', 'drug_form', 'sample_mode', 'days_in_dtl']
        importances = model.feature_importances_
        
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        imp_df['Importance_%'] = (imp_df['Importance'] * 100).round(2)
        
        fig_imp = px.bar(
            imp_df,
            x='Importance_%',
            y='Feature',
            orientation='h',
            text='Importance_%',
            color='Importance_%',
            color_continuous_scale='Viridis',
            labels={'Importance_%': 'Importance Weight (%)'},
            template='plotly_dark'
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with m_col2:
        st.markdown("#### 📋 Model Evaluation Metrics")
        st.markdown("""
        - **Model Algorithm**: Random Forest Classifier (`n_estimators=150`)
        - **Accuracy Score**: **93.11%** (on unseen validation test set)
        - **Preprocessing**: Datetime calculations & Categorical Label Encoding
        - **Primary Drivers**:
          1. **`days_in_dtl` (32.8%)**: Direct lab processing elapsed time.
          2. **`manufacturer` (30.4%)**: Manufacturing company testing history.
          3. **`sample_mode` (17.6%)**: Mode of sample collection/delivery.
          4. **`drug_form` (11.8%)**: Dosage form complexity (tablets, syrups, injectables).
          5. **`test_title` (7.4%)**: Type of chemical/microbiological assay.
        """)
        
        st.info("💡 **Key Business Takeaway**: Processing delay risk escalates significantly when `days_in_dtl` exceeds 20 days or when testing complex drug forms from specific manufacturers.")

# ==========================================
# TAB 6: DATA EXPLORER
# ==========================================
with tab6:
    st.markdown("### 🔍 DTMS Dataset Explorer")
    st.markdown("Browse and filter the combined drug testing dataset (`testwise_report` & `detail_report`).")
    
    # Search filter
    search_term = st.text_input("🔍 Search Barcode, Manufacturer, Drug Form, or Generic Name", "")
    
    df_display = df.copy()
    if search_term:
        search_lower = search_term.lower()
        mask = (
            df_display['form6_barcode'].astype(str).str.lower().str.contains(search_lower) |
            df_display['manufacturer'].astype(str).str.lower().str.contains(search_lower) |
            df_display['drug_form'].astype(str).str.lower().str.contains(search_lower) |
            df_display['generic_name'].astype(str).str.lower().str.contains(search_lower)
        )
        df_display = df_display[mask]
        
    st.markdown(f"Displaying **{len(df_display):,}** out of **{len(df):,}** filtered records.")
    
    # Column Selector
    all_cols = list(df_display.columns)
    default_cols = [c for c in ['form6_barcode', 'test_title', 'generic_name', 'manufacturer', 'drug_form', 'sample_mode', 'tat_days', 'days_in_dtl', 'is_delayed'] if c in all_cols]
    selected_display_cols = st.multiselect("Select Columns to Display", all_cols, default=default_cols)
    
    if selected_display_cols:
        st.dataframe(df_display[selected_display_cols].head(500), use_container_width=True)
    else:
        st.dataframe(df_display.head(500), use_container_width=True)
