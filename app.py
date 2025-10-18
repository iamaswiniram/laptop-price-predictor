# =============================================================================
# FILE: app.py
# PURPOSE: The Streamlit web application for predicting laptop prices.
# This file MUST exist in the same directory as your notebook when you save the model.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# =============================================================================
# 1. DEFINE THE CUSTOM CLASSES (THE "FUTURE ADDRESS" FOR THE SAVED MODEL)
# =============================================================================
class ColumnAs2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.to_numpy().reshape(-1, 1)

class DataFrameMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self): self.binarizers, self.feature_names_ = {}, []
    def fit(self, X, y=None):
        self.feature_names_ = []
        for col in X.columns:
            mlb = MultiLabelBinarizer(sparse_output=False)
            mlb.fit(X[col])
            self.binarizers[col] = mlb
            self.feature_names_.extend([f"{col}_{cls}" for cls in mlb.classes_])
        return self
    def transform(self, X):
        all_transformed = [self.binarizers[col].transform(X[col]) for col in X.columns]
        return np.hstack(all_transformed)
    def get_feature_names_out(self, input_features=None): return np.array(self.feature_names_, dtype=object)

# =============================================================================
# 2. DEFINE THE FEATURE ENGINEERING FUNCTION
# =============================================================================
def engineer_features(df):
    df_copy = df.copy()
    for col in ['CPU cores', 'CPU clock speed (GHz)', 'drive memory size (GB)']:
        if col in df_copy.columns: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    if 'CPU model' in df_copy.columns: df_copy['CPU model'] = df_copy['CPU model'].astype(str)
    for col in ['communications', 'multimedia', 'input devices', 'operating system']:
        if col in df_copy.columns: df_copy[col] = df_copy[col].apply(lambda x: x if isinstance(x, list) else [])
    df_copy['RAM_size_GB'] = df_copy['RAM size'].str.extract('(\d+)').astype(float)
    res_split = df_copy['resolution (px)'].str.split(' x ', expand=True)
    df_copy['pixel_count'] = pd.to_numeric(res_split[0], errors='coerce') * pd.to_numeric(res_split[1], errors='coerce')
    df_copy['screen_size_inch'] = df_copy['screen size'].str.extract('(\d+\.?\d*)').astype(float)
    df_copy['is_ssd'] = df_copy['drive type'].str.contains('ssd', case=False, na=False).astype(int)
    df_copy['is_hdd'] = df_copy['drive type'].str.contains('hdd', case=False, na=False).astype(int)
    df_copy['has_windows'] = df_copy['operating system'].apply(lambda x: 1 if any('windows' in s.lower() for s in x) else 0)
    return df_copy

# =============================================================================
# 3. LOAD ARTIFACTS AND RUN THE APP
# =============================================================================
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Prediction App")
st.markdown("Enter the laptop's specifications below to get an estimated price.")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('lgbm_price_predictor.joblib')
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
    st.success("Model and preprocessor loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts. Please ensure 'preprocessor.joblib' and 'lgbm_price_predictor.joblib' are present. Error: {e}")
    st.stop()

# --- User Input Form ---
with st.form("prediction_form"):
    # ... (Your full UI code goes here) ...
    st.header("Enter Laptop Specifications")
    col1, col2, col3 = st.columns(3)
    with col1:
        ram_size = st.selectbox("RAM Size", ['16 gb', '8 gb', '32 gb'])
        cpu_model = st.selectbox("CPU Model", ['Intel Core i7', 'Intel Core i5', 'AMD Ryzen 7'])
    with col2:
        drive_memory_size = st.number_input("Drive Memory (GB)", 128, 4096, 512)
        drive_type = st.selectbox("Drive Type", ['SSD', 'HDD', 'SSD + HDD'])
    with col3:
        resolution = st.selectbox("Resolution", ['1920 x 1080', '2560 x 1440'])
        screen_size = st.text_input("Screen Size (e.g., 15.6 inch)", "15.6 inch")
    
    # Simplified for example, add all your other inputs
    graphic_card_type = 'integrated'
    ram_type = 'DDR4'
    state = 'new'
    warranty = 'manufacturer'
    cpu_cores = 8
    cpu_clock_speed = 2.5
    communications = ['Bluetooth', 'Wi-Fi']
    multimedia = ['camera', 'speakers']
    input_devices = ['keyboard', 'touchpad']
    operating_system = ['Windows 11 Home']
    
    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = {
        'CPU cores': cpu_cores, 'CPU clock speed (GHz)': cpu_clock_speed, 'drive memory size (GB)': drive_memory_size,
        'RAM size': ram_size, 'resolution (px)': resolution, 'screen size': screen_size,
        'graphic card type': graphic_card_type, 'RAM type': ram_type, 'state': state,
        'warranty': warranty, 'CPU model': cpu_model, 'communications': communications,
        'multimedia': multimedia, 'input devices': input_devices, 'operating system': operating_system,
        'drive type': drive_type
    }
    input_df = pd.DataFrame([input_data])
    
    with st.spinner('Analyzing specifications and predicting price...'):
        featured_df = engineer_features(input_df)
        processed_input = preprocessor.transform(featured_df)
        log_prediction = model.predict(processed_input)
        final_prediction = np.expm1(log_prediction[0])

    st.success("Prediction Complete!")
    st.metric(label="Predicted Laptop Price", value=f"${final_prediction:,.2f}")
    st.balloons()
