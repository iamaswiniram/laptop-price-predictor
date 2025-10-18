import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import sys, types, re

# Ensure working directory
try:
    os.chdir(os.path.dirname(__file__))
except Exception:
    pass

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .sidebar .sidebar-content { background: #f0f2f6; }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius:10px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    .stButton>button:hover { background-color: #45a049; }
    .st-expander { border: 1px solid #ddd; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Custom transformer stubs (simplified) – replace with true implementations if you retrain
# ---------------------------------------------------------------------
class ColumnAs2D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.to_numpy().reshape(-1, 1)

class DataFrameMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binarizers = {}
        self.feature_names_ = []
    def fit(self, X, y=None): return self
    def transform(self, X):
        # Will fail later if expected binarizers missing – indicates stub vs real mismatch
        all_transformed = [self.binarizers[col].transform(X[col]) for col in X.columns]
        return np.hstack(all_transformed)
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)

def _inject_stub_classes():
    # Add stubs under possible training module names
    for mod_name in [
        'custom_transformers',
        'transformers',
        'preprocessing',
        'ml_utils',
        '__main__'
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.ColumnAs2D = ColumnAs2D
            m.DataFrameMultiLabelBinarizer = DataFrameMultiLabelBinarizer
            sys.modules[mod_name] = m

# ---------------------------------------------------------------------
# Debug helper: scan pickle file bytes to display probable module paths
# ---------------------------------------------------------------------
def scan_pickle_modules(path):
    try:
        raw = open(path, 'rb').read()
    except Exception:
        return []
    # Heuristic: look for patterns like b'module.name\nClassName'
    candidates = set()
    for cls in [b'ColumnAs2D', b'DataFrameMultiLabelBinarizer']:
        # module name often precedes class with newline
        pattern = rb'([a-zA-Z_][\w\.]*)\n' + cls
        for m in re.finditer(pattern, raw):
            candidates.add(m.group(1).decode('utf-8'))
    return sorted(candidates)

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def engineer_features(df):
    df_copy = df.copy()
    for col in ['CPU cores', 'CPU clock speed (GHz)', 'drive memory size (GB)']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    if 'CPU model' in df_copy.columns:
        df_copy['CPU model'] = df_copy['CPU model'].astype(str)
    for col in ['communications', 'multimedia', 'input devices', 'operating system']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(lambda x: x if isinstance(x, list) else [])
    if 'RAM size' in df_copy.columns:
        df_copy['RAM_size_GB'] = df_copy['RAM size'].str.extract(r'(\d+)').astype(float)
    if 'resolution (px)' in df_copy.columns:
        res_split = df_copy['resolution (px)'].str.split(' x ', expand=True)
        df_copy['pixel_count'] = (
            pd.to_numeric(res_split[0], errors='coerce') *
            pd.to_numeric(res_split[1], errors='coerce')
        )
    if 'screen size' in df_copy.columns:
        df_copy['screen_size_inch'] = df_copy['screen size'].str.extract(r'(\d+\.?\d*)').astype(float)
    if 'drive type' in df_copy.columns:
        df_copy['is_ssd'] = df_copy['drive type'].str.contains('ssd', case=False, na=False).astype(int)
        df_copy['is_hdd'] = df_copy['drive type'].str.contains('hdd', case=False, na=False).astype(int)
    if 'operating system' in df_copy.columns:
        df_copy['has_windows'] = df_copy['operating system'].apply(
            lambda x: 1 if any(isinstance(s, str) and 'windows' in s.lower() for s in x) else 0
        )
    df_copy = df_copy.drop(
        columns=['RAM size', 'resolution (px)', 'screen size', 'drive type', 'operating system'],
        errors='ignore'
    )
    return df_copy

# ---------------------------------------------------------------------
# Load model & preprocessor
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    _inject_stub_classes()
    model = None
    preprocessor = None

    # Show probable module names in sidebar for user guidance
    with st.sidebar:
        mods = scan_pickle_modules('preprocessor.joblib')
        if mods:
            st.caption("Detected module name candidates for custom classes:")
            for m in mods:
                st.code(m)
        else:
            st.caption("No module name candidates found via byte scan.")

    try:
        model = joblib.load('lgbm_price_predictor.joblib')
    except FileNotFoundError:
        st.error("Missing lgbm_price_predictor.joblib.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    try:
        preprocessor = joblib.load('preprocessor.joblib')
    except FileNotFoundError:
        st.error("Missing preprocessor.joblib.")
        return None, None
    except AttributeError as e:
        st.error("Custom class/module mismatch. Retrain OR create the original module file with real class code.")
        return model, None
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return model, None

    return model, preprocessor

model, preprocessor = load_model()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("laptop_image.jpg", width=250)
    except Exception:
        st.write("Image not found.")
with col2:
    st.title("Laptop Price Predictor")
    st.markdown("Enter the specifications of a laptop to estimate its price.")

st.markdown("---")

if model is not None and preprocessor is not None:
    with st.form("prediction_form"):
        st.header("Enter Laptop Specifications")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.subheader("Core Components")
            cpu_model = st.selectbox("CPU Model", [
                'intel core i7', 'intel core i5', 'amd ryzen 7', 'amd ryzen 5',
                'intel core i9', 'intel core i3', 'amd ryzen 9', 'amd ryzen 3'
            ])
            cpu_cores = st.slider("CPU Cores", 2, 16, 4)
            cpu_clock_speed = st.slider("CPU Clock Speed (GHz)", 1.0, 5.0, 2.5, 0.1)
            ram_size = st.selectbox("RAM Size", ['8 gb', '16 gb', '32 gb', '4 gb', '64 gb', '12 gb'])
            ram_type = st.selectbox("RAM Type", ['ddr4', 'ddr5', 'ddr3'])

        with col_b:
            st.subheader("Storage & Graphics")
            drive_type = st.selectbox("Drive Type", ['ssd', 'ssd + hdd', 'hdd'])
            drive_memory_size = st.number_input("Drive Memory Size (GB)", min_value=128, max_value=4096, value=512, step=128)
            graphic_card_type = st.selectbox("Graphic Card Type", ['integrated graphics', 'dedicated graphics'])

        with col_c:
            st.subheader("Display & Condition")
            screen_size = st.selectbox("Screen Size", [
                '15" - 15.9"', '14" - 14.9"', '13" - 13.9"', '16" - 16.9"', '17" or more'
            ])
            resolution = st.selectbox("Resolution (px)", [
                '1920 x 1080', '2560 x 1440', '1366 x 768', '3840 x 2160'
            ])
            state = st.selectbox("Condition", ['new', 'used', 'refurbished', 'after-exhibition'])

        with st.expander("Additional Features"):
            communications = st.multiselect("Communications", [
                'bluetooth', 'wifi', 'lan 10/100/1000 mbps', 'nfc'
            ], default=['bluetooth', 'wifi'])
            multimedia = st.multiselect("Multimedia", ['camera', 'speakers', 'microphone'],
                                        default=['camera', 'speakers', 'microphone'])
            input_devices = st.multiselect("Input Devices", [
                'keyboard', 'touchpad', 'numeric keyboard', 'illuminated keyboard'
            ], default=['keyboard', 'touchpad'])
            operating_system = st.multiselect("Operating System", [
                'windows 11 home', 'windows 10 home', 'no system', 'macos'
            ], default=['windows 11 home'])
            warranty = st.selectbox("Warranty", ['producer warranty', 'seller warranty', 'no warranty'])

        st.markdown("---")
        submitted = st.form_submit_button("Predict Price")

    if submitted:
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
        input_df = pd.DataFrame([input_data])

        with st.spinner('Analyzing specifications and predicting price...'):
            try:
                featured_df = engineer_features(input_df)
                processed = preprocessor.transform(featured_df)
                log_pred = model.predict(processed)
                final_price = np.expm1(log_pred[0])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.success("Prediction Complete!")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; text-align: center;">
            <h2 style="color: #2E8B57;">Estimated Laptop Price</h2>
            <h1 style="color: #4CAF50; font-size: 3em;">${final_price:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
elif model is not None and preprocessor is None:
    st.error("Preprocessor failed to load. See sidebar for module hints. Option: retrain with classes in a module (e.g. custom_transformers.py).")
else:
    st.warning("Model not loaded. Fix earlier errors before using the form.")

st.markdown("---")
st.markdown("Developed by a Machine Learning enthusiast.")
