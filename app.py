# =============================================================================
# FILE: app.py
# PURPOSE: The "Self-Healing" Streamlit App. It trains the model itself
# if the model files are not found, guaranteeing a perfect environment match.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

# --- Configuration ---
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
PREPROCESSOR_PATH = 'preprocessor.joblib'
MODEL_PATH = 'lgbm_price_predictor.joblib'

# =============================================================================
# 1. DEFINE THE CUSTOM CLASSES AND FUNCTIONS (NEEDED FOR TRAINING & LOADING)
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
    df_copy = df_copy.drop(columns=['RAM size', 'resolution (px)', 'screen size', 'drive type', 'operating system'], errors='ignore')
    return df_copy

# =============================================================================
# 2. THE TRAINING FUNCTION
# This function will run ONLY ONCE when the app first starts on the server.
# =============================================================================
@st.cache_resource(show_spinner="First-time setup: Training model, please wait...")
def train_and_save_artifacts():
    """
    Loads data, trains the model, and saves the artifacts on the server.
    This is cached, so it only runs once.
    """
    # --- Data Loading ---
    train_df = pd.read_json('train_dataset.json', orient='columns')
    val_df = pd.read_json('val_dataset.json', orient='columns')

    # --- Feature Engineering & Final Data Prep ---
    train_featured_df = engineer_features(train_df)
    val_featured_df = engineer_features(val_df)
    full_training_data = pd.concat([train_featured_df, val_featured_df], ignore_index=True)
    
    y_final_train = np.log1p(full_training_data['buynow_price'])
    X_final_train = full_training_data.drop(columns=['buynow_price', 'log_buynow_price'], errors='ignore')

    # --- Define Preprocessing Pipeline ---
    NUMERIC_FEATURES = ['CPU cores', 'CPU clock speed (GHz)', 'drive memory size (GB)', 'RAM_size_GB', 'pixel_count', 'screen_size_inch']
    CATEGORICAL_FEATURES_LOW = ['graphic card type', 'RAM type', 'state', 'warranty']
    BOOLEAN_FEATURES = ['is_ssd', 'is_hdd', 'has_windows']
    CATEGORICAL_FEATURES_HIGH = ['CPU model']
    MULTI_LABEL_FEATURES = ['communications', 'multimedia', 'input devices']

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer_low = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    categorical_transformer_high = Pipeline(steps=[('to_2d', ColumnAs2D()), ('hasher', FeatureHasher(n_features=10, input_type='string'))])
    multi_label_transformer = Pipeline(steps=[('mlb', DataFrameMultiLabelBinarizer())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat_low', categorical_transformer_low, CATEGORICAL_FEATURES_LOW),
            ('cat_high', categorical_transformer_high, CATEGORICAL_FEATURES_HIGH),
            ('multi_label', multi_label_transformer, MULTI_LABEL_FEATURES),
            ('passthrough_bool', 'passthrough', BOOLEAN_FEATURES)
        ],
        remainder='drop'
    )

    # --- Fit the preprocessor and transform the data ---
    X_final_processed = preprocessor.fit_transform(X_final_train)

    # --- Define and train the final model ---
    best_hyperparameters = {
        'n_estimators': 800, 'reg_lambda': 0.1, 'reg_alpha': 0.1, 
        'num_leaves': 60, 'learning_rate': 0.05
    }
    final_model = lgb.LGBMRegressor(**best_hyperparameters, random_state=RANDOM_STATE)
    final_model.fit(X_final_processed, y_final_train)

    # --- Save the artifacts ---
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    return preprocessor, final_model

# =============================================================================
# 3. MAIN APP LOGIC
# =============================================================================
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Prediction App")
st.markdown("Enter the laptop's specifications below to get an estimated price.")

# Check if model files exist. If not, train them.
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        preprocessor, model = train_and_save_artifacts()
    else:
        # If they exist, just load them.
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
    st.success("Model is ready!")
except Exception as e:
    st.error(f"An error occurred. Please try rebooting the app. Error details: {e}")
    st.stop()


# --- User Input Form ---
with st.form("prediction_form"):
    st.header("Enter Laptop Specifications")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Core Components")
        cpu_model = st.selectbox("CPU Model", ['Intel Core i7', 'Intel Core i5', 'AMD Ryzen 7', 'AMD Ryzen 5', 'Intel Core i9', 'Intel Core i3', 'AMD Ryzen 9', 'AMD Ryzen 3'])
        cpu_cores = st.slider("CPU Cores", 2, 32, 8)
        cpu_clock_speed = st.slider("CPU Clock Speed (GHz)", 1.0, 5.5, 2.8, 0.1)
        ram_size = st.selectbox("RAM Size", ['16 gb', '8 gb', '32 gb', '4 gb', '64 gb', '12 gb'])
        ram_type = st.selectbox("RAM Type", ['DDR4', 'DDR5', 'LPDDR4X', 'LPDDR5', 'DDR3'])

    with col2:
        st.subheader("Storage & Graphics")
        drive_type = st.selectbox("Drive Type", ['SSD', 'SSD + HDD', 'HDD'])
        drive_memory_size = st.number_input("Drive Memory Size (GB)", min_value=128, max_value=8192, value=512, step=128)
        graphic_card_type = st.selectbox("Graphic Card Type", ['integrated', 'dedicated'])
        
    with col3:
        st.subheader("Display & Condition")
        screen_size = st.text_input("Screen Size (e.g., 15.6 inch)", "15.6 inch")
        resolution = st.selectbox("Resolution (px)", ['1920 x 1080', '2560 x 1440', '1366 x 768', '3840 x 2160', '3072 x 1920'])
        state = st.selectbox("Condition", ['new', 'used', 'manufacturer refurbished', 'seller refurbished'])
        warranty = st.selectbox("Warranty", ['manufacturer', 'seller', 'no warranty'])

    with st.expander("Additional Features"):
        communications = st.multiselect("Communications", ['Bluetooth', 'Wi-Fi', 'LAN', 'NFC'], default=['Bluetooth', 'Wi-Fi'])
        multimedia = st.multiselect("Multimedia", ['camera', 'speakers', 'microphone'], default=['camera', 'speakers', 'microphone'])
        input_devices = st.multiselect("Input Devices", ['keyboard', 'touchpad', 'backlit keyboard', 'numeric keyboard'], default=['keyboard', 'touchpad'])
        operating_system = st.multiselect("Operating System", ['Windows 11 Home', 'Windows 10 Pro', 'No OS', 'macOS'], default=['Windows 11 Home'])

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
