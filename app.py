import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer # <-- IMPORTANT: Import this

# =============================================================================
# Page Configuration & Title
# =============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Rich UI ---
# (Your CSS is great, no changes needed here)
st.markdown("""
<style>
    /* ... your CSS ... */
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions & Classes (Must match the ones used in training)
# =============================================================================

class ColumnAs2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.to_numpy().reshape(-1, 1)

# =============================================================================
# FIXED SECTION 1: Corrected DataFrameMultiLabelBinarizer
# This class definition must exist so joblib can load the preprocessor,
# but its 'fit' method is only used during training, not in the app.
# We revert it to the original, correct version from the training script.
# =============================================================================
class DataFrameMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binarizers = {}
        self.feature_names_ = []

    def fit(self, X, y=None):
        self.feature_names_ = []
        for col in X.columns:
            # This is the correct logic: create a new binarizer and fit it
            mlb = MultiLabelBinarizer(sparse_output=False)
            mlb.fit(X[col])
            self.binarizers[col] = mlb
            self.feature_names_.extend([f"{col}_{cls}" for cls in mlb.classes_])
        return self

    def transform(self, X):
        all_transformed = []
        for col in X.columns:
            # Use the binarizer that was stored during the .fit() call
            all_transformed.append(self.binarizers[col].transform(X[col]))
        return np.hstack(all_transformed)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


def engineer_features(df):
    df_copy = df.copy()
    for col in ['CPU cores', 'CPU clock speed (GHz)', 'drive memory size (GB)']:
        if col in df_copy.columns: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    if 'CPU model' in df_copy.columns: df_copy['CPU model'] = df_copy['CPU model'].astype(str)
    for col in ['communications', 'multimedia', 'input devices', 'operating system']:
        if col in df_copy.columns: df_copy[col] = df_copy[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # Parsing features
    df_copy['RAM_size_GB'] = df_copy['RAM size'].str.extract('(\d+)').astype(float)
    res_split = df_copy['resolution (px)'].str.split(' x ', expand=True)
    df_copy['pixel_count'] = pd.to_numeric(res_split[0], errors='coerce') * pd.to_numeric(res_split[1], errors='coerce')
    df_copy['screen_size_inch'] = df_copy['screen size'].str.extract('(\d+\.?\d*)').astype(float)
    df_copy['is_ssd'] = df_copy['drive type'].str.contains('ssd', case=False, na=False).astype(int)
    df_copy['is_hdd'] = df_copy['drive type'].str.contains('hdd', case=False, na=False).astype(int)
    
    # =============================================================================
    # FIXED SECTION 2: Made the 'windows' check case-insensitive for robustness
    # =============================================================================
    df_copy['has_windows'] = df_copy['operating system'].apply(lambda x: 1 if any('windows' in s.lower() for s in x) else 0)
    
    # We don't need to drop columns here for the app, but it doesn't hurt
    df_copy = df_copy.drop(columns=['RAM size', 'resolution (px)', 'screen size', 'drive type', 'operating system'], errors='ignore')
    return df_copy

# =============================================================================
# Load Model and Preprocessor
# =============================================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('lgbm_price_predictor.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor files not found. Make sure 'lgbm_price_predictor.joblib' and 'preprocessor.joblib' are in the same directory.")
        return None, None

model, preprocessor = load_model()

# =============================================================================
# Application UI (Your UI code is excellent, no changes needed)
# =============================================================================
# --- Header ---
# Using a placeholder image path. Make sure you have an image named 'laptop_image.jpg'
# or change the path. I'll add a check to prevent an error if it's missing.
try:
    st.image("laptop_image.jpg", width=150)
except Exception:
    st.info("Info: 'laptop_image.jpg' not found. You can add an image to the directory.")

st.title("Laptop Price Predictor")
st.markdown("Enter the specifications of a laptop, and our machine learning model will estimate its price.")

st.markdown("---")

# --- Input Form ---
if model is not None:
    with st.form("prediction_form"):
        st.header("Enter Laptop Specifications")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Core Components")
            cpu_model = st.selectbox("CPU Model", ['intel core i7', 'intel core i5', 'amd ryzen 7', 'amd ryzen 5', 'intel core i9', 'intel core i3', 'amd ryzen 9', 'amd ryzen 3'])
            cpu_cores = st.slider("CPU Cores", 2, 16, 8)
            cpu_clock_speed = st.slider("CPU Clock Speed (GHz)", 1.0, 5.0, 2.8, 0.1)
            ram_size = st.selectbox("RAM Size", ['16 gb', '8 gb', '32 gb', '4 gb', '64 gb', '12 gb'])
            ram_type = st.selectbox("RAM Type", ['DDR4', 'DDR5', 'LPDDR4X', 'LPDDR5'])

        with col2:
            st.subheader("Storage & Graphics")
            drive_type = st.selectbox("Drive Type", ['SSD', 'SSD + HDD', 'HDD'])
            drive_memory_size = st.number_input("Drive Memory Size (GB)", min_value=128, max_value=4096, value=512, step=128)
            graphic_card_type = st.selectbox("Graphic Card Type", ['integrated', 'dedicated'])
            
        with col3:
            st.subheader("Display & Condition")
            screen_size = st.selectbox("Screen Size", ['15.6 inch', '14 inch', '13.3 inch', '16 inch', '17.3 inch'])
            resolution = st.selectbox("Resolution (px)", ['1920 x 1080', '2560 x 1440', '1366 x 768', '3840 x 2160'])
            state = st.selectbox("Condition", ['new', 'used', 'manufacturer refurbished', 'seller refurbished'])

        with st.expander("Additional Features (Connectivity, Multimedia, etc.)"):
            communications = st.multiselect("Communications", ['Bluetooth', 'Wi-Fi', 'LAN'], default=['Bluetooth', 'Wi-Fi'])
            multimedia = st.multiselect("Multimedia", ['camera', 'speakers', 'microphone'], default=['camera', 'speakers', 'microphone'])
            input_devices = st.multiselect("Input Devices", ['keyboard', 'touchpad', 'backlit keyboard'], default=['keyboard', 'touchpad'])
            operating_system = st.multiselect("Operating System", ['Windows 11 Home', 'Windows 10 Pro', 'No OS', 'macOS'], default=['Windows 11 Home'])
            warranty = st.selectbox("Warranty", ['manufacturer', 'seller'])

        st.markdown("---")
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        # The multiselect widgets return a list, which is what we need.
        # For single-row DataFrame creation, we need to wrap the list in another list
        # for the multi-label columns. However, pd.DataFrame([dict]) handles this correctly.
        input_data = {
            'graphic card type': graphic_card_type,
            'communications': communications,
            'resolution (px)': resolution,
            'CPU cores': cpu_cores,
            'RAM size': ram_size,
            'operating system': operating_system,
            'drive type': drive_type,
            'input devices': input_devices,
            'multimedia': multimedia,
            'RAM type': ram_type,
            'CPU clock speed (GHz)': cpu_clock_speed,
            'CPU model': cpu_model,
            'state': state,
            'drive memory size (GB)': drive_memory_size,
            'warranty': warranty,
            'screen size': screen_size
        }

        # Convert to a single-row DataFrame
        input_df = pd.DataFrame([input_data])

        with st.spinner('Analyzing specifications and predicting price...'):
            featured_df = engineer_features(input_df)
            processed_df = preprocessor.transform(featured_df)
            log_prediction = model.predict(processed_df)
            final_prediction = np.expm1(log_prediction[0])

        st.success("Prediction Complete!")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #2E8B57;">Estimated Laptop Price</h2>
            <h1 style="color: #4CAF50; font-size: 3em;">${final_prediction:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

st.markdown("---")
st.markdown("Developed by a Machine Learning enthusiast.")

