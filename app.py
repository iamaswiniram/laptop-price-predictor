# =============================================================================
# FILE: app.py
# PURPOSE: The "Diagnostic & Control" App. This version includes diagnostics
# and a force-retrain button to defeat all environment and caching issues.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sklearn
import lightgbm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin

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
# =============================================================================
def train_and_save_artifacts():
    """
    This function is the core of the self-healing process.
    """
    with st.spinner("First-time setup: Training model, please wait a moment..."):
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
        final_model = lightgbm.LGBMRegressor(**best_hyperparameters, random_state=RANDOM_STATE)
        final_model.fit(X_final_processed, y_final_train)

        # --- Save the artifacts ---
        joblib.dump(final_model, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    st.success("Model training complete and artifacts saved!")
    st.balloons()
    return preprocessor, final_model

# =============================================================================
# 3. MAIN APP LOGIC
# =============================================================================
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Prediction App")

# --- DIAGNOSTICS EXPANDER ---
with st.expander("Show Environment Diagnostics"):
    st.write("This shows the actual library versions installed on the server.")
    st.write(f"Pandas version: {pd.__version__}")
    st.write(f"NumPy version: {np.__version__}")
    st.write(f"Scikit-learn version: {sklearn.__version__}")
    st.write(f"LightGBM version: {lightgbm.__version__}")
    try:
        import catboost
        st.write(f"CatBoost version: {catboost.__version__}")
    except ImportError:
        st.error("CatBoost is NOT installed.")

# --- CONTROL PANEL IN SIDEBAR ---
st.sidebar.title("Control Panel")
if st.sidebar.button("Force Retrain Model", help="Click this if the app is crashing. It will delete the saved model files and train a new one from scratch."):
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(PREPROCESSOR_PATH):
        os.remove(PREPROCESSOR_PATH)
    st.sidebar.success("Cache cleared. The app will now retrain the model on the next run.")
    st.experimental_rerun()

# --- MODEL LOADING LOGIC ---
preprocessor = None
model = None
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        preprocessor, model = train_and_save_artifacts()
    else:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"A critical error occurred while loading or training the model. Please click the 'Force Retrain Model' button in the sidebar. Error details: {e}")
    st.stop()

if preprocessor and model:
    st.success("Model is ready!")
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
