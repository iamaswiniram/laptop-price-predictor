# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# =============================================================================
# Helper Classes & Functions (Must be defined for the preprocessor to load)
# =============================================================================
# These classes and functions are copied directly from your training script.
# They are required for joblib to correctly load the preprocessor object.

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
    # The drop columns are not needed here as we build the input from scratch
    return df_copy

# =============================================================================
# Load Model and Preprocessor
# =============================================================================
# Use st.cache_resource to load these only once
@st.cache_resource
def load_artifacts():
    """
    Loads the saved model and preprocessor from disk.
    Caches the result to avoid reloading on every interaction.
    """
    try:
        model = joblib.load('lgbm_price_predictor.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Make sure 'lgbm_price_predictor.joblib' and 'preprocessor.joblib' are in the same directory as app.py.")
        return None, None

model, preprocessor = load_artifacts()

# =============================================================================
# Main App Interface
# =============================================================================
st.title("💻 Laptop Price Prediction App")
st.markdown("""
This application uses a LightGBM machine learning model to predict the price of a laptop.
Enter the laptop's specifications in the sidebar to get an estimated price.
""")

st.sidebar.header("Enter Laptop Specifications")

# --- Create Input Widgets in the Sidebar ---
def user_input_features():
    # Numerical Inputs
    cpu_cores = st.sidebar.number_input('CPU Cores', min_value=1, max_value=32, value=8, step=1)
    cpu_clock = st.sidebar.number_input('CPU Clock Speed (GHz)', min_value=0.5, max_value=6.0, value=2.5, step=0.1)
    drive_memory = st.sidebar.number_input('Drive Memory Size (GB)', min_value=64, max_value=4096, value=512, step=64)
    
    # Text inputs that get parsed
    ram_size = st.sidebar.text_input("RAM Size (e.g., '16 gb')", value='16 gb')
    resolution = st.sidebar.text_input("Screen Resolution (e.g., '1920 x 1080')", value='1920 x 1080')
    screen_size = st.sidebar.text_input("Screen Size (e.g., '15.6 inch')", value='15.6 inch')
    drive_type = st.sidebar.text_input("Drive Type (e.g., 'SSD', 'HDD + SSD')", value='SSD')
    cpu_model = st.sidebar.text_input("CPU Model (e.g., 'Intel Core i7')", value='Intel Core i7')

    # Low-cardinality categorical inputs
    graphic_card_type = st.sidebar.selectbox('Graphic Card Type', ['integrated', 'dedicated'])
    ram_type = st.sidebar.selectbox('RAM Type', ['DDR4', 'DDR5', 'LPDDR4X', 'LPDDR5'])
    state = st.sidebar.selectbox('Condition', ['new', 'used', 'manufacturer refurbished', 'seller refurbished'])
    warranty = st.sidebar.selectbox('Warranty', ['manufacturer', 'seller'])

    # Multi-label inputs
    communications = st.sidebar.multiselect('Communications', ['Bluetooth', 'Wi-Fi', 'LAN'], default=['Bluetooth', 'Wi-Fi'])
    multimedia = st.sidebar.multiselect('Multimedia', ['camera', 'speakers', 'microphone'], default=['camera', 'speakers', 'microphone'])
    input_devices = st.sidebar.multiselect('Input Devices', ['keyboard', 'touchpad', 'backlit keyboard'], default=['keyboard', 'touchpad'])
    operating_system = st.sidebar.multiselect('Operating System', ['Windows 11 Home', 'Windows 10 Pro', 'No OS', 'macOS'], default=['Windows 11 Home'])

    # Create a dictionary from the inputs
    data = {
        'CPU cores': cpu_cores,
        'CPU clock speed (GHz)': cpu_clock,
        'drive memory size (GB)': drive_memory,
        'RAM size': ram_size,
        'resolution (px)': resolution,
        'screen size': screen_size,
        'drive type': drive_type,
        'CPU model': cpu_model,
        'graphic card type': graphic_card_type,
        'RAM type': ram_type,
        'state': state,
        'warranty': warranty,
        'communications': [communications], # Must be in a list for DataFrame creation
        'multimedia': [multimedia],
        'input devices': [input_devices],
        'operating system': [operating_system]
    }
    
    # Convert to a single-row DataFrame
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Display the user's input
st.subheader("Your Laptop's Specifications")
st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# Prediction button
if st.button('Predict Price', type="primary"):
    if model is not None and preprocessor is not None:
        with st.spinner('Analyzing specifications and predicting price...'):
            # 1. Apply feature engineering
            engineered_df = engineer_features(input_df)
            
            # 2. Apply the preprocessor
            # Ensure the columns are in the same order as during training
            # The preprocessor will select the correct ones
            processed_input = preprocessor.transform(engineered_df)
            
            # 3. Make a prediction (output is log-transformed)
            log_prediction = model.predict(processed_input)
            
            # 4. Inverse transform the prediction to get the actual price
            predicted_price = np.expm1(log_prediction)[0]

        # Display the result
        st.success("Prediction Complete!")
        st.metric(label="Predicted Laptop Price", value=f"${predicted_price:,.2f}")
        st.balloons()
    else:
        st.error("Model is not loaded. Cannot make a prediction.")
