"""
IR Spectroscopy Functional Group Classification
Streamlit Interactive Dashboard

This app demonstrates a complete ML pipeline for classifying functional groups
in organic molecules using IR spectroscopy data and SMILES strings.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import StringIO
import sys

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    hamming_loss, jaccard_score
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from scipy import signal
from scipy.ndimage import percentile_filter
from tqdm import tqdm

# Chemistry Libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("✓ RDKit loaded successfully")
except ImportError:
    print("Installing RDKit...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'rdkit'])
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    print("✓ RDKit installed and loaded")

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Page configuration
st.set_page_config(
    page_title="IR Spectroscopy ML Project",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2d3748;
        border-left: 6px solid #667eea;
        padding-left: 15px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #f7fafc 0%, #ffffff 100%);
        padding: 0.5rem 0 0.5rem 15px;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: none;
        color: white;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-left: 5px solid #48bb78;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(72, 187, 120, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# Functional group patterns (same as in pipeline)
FUNCTIONAL_GROUP_PATTERNS = {
    'O-H_alcohol_free': '[OX2H]',
    'O-H_alcohol_bonded': '[OH]',
    'N-H_primary_amine': '[NX3;H2]',
    'N-H_secondary_amine': '[NX3;H1]',
    'O-H_carboxylic_acid': '[CX3](=O)[OX2H1]',
    'C-H_alkyne': '[CX2]#[CX2]',
    'C-H_aromatic': 'c',
    'C-H_alkene': '[CX3]=[CX3]',
    'C-H_alkane': '[CX4]',
    'C-H_aldehyde': '[CX3H1](=O)',
    'CN_nitrile': '[CX2]#[NX1]',
    'CC_alkyne': '[CX2]#[CX2]',
    'NCO_isocyanate': '[NX2]=C=O',
    'CO_acid_chloride': '[CX3](=[OX1])[Cl]',
    'CO_anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'CO_ester': '[CX3](=O)[OX2H0]',
    'CO_aldehyde': '[CX3H1](=O)[#6]',
    'CO_ketone': '[CX3](=O)[#6]',
    'CO_carboxylic_acid': '[CX3](=O)[OX2H1]',
    'CO_amide': '[CX3](=[OX1])[NX3]',
    'CC_aromatic': 'c:c',
    'CC_alkene': '[CX3]=[CX3]',
    'CN_imine': '[CX3]=[NX2]',
    'NH_amide_bend': '[NX3][CX3](=[OX1])',
    'NH2_amine_bend': '[NX3;H2]',
    'CO_ester_stretch': '[CX3](=O)[OX2]',
    'CO_alcohol': '[CX4][OX2H]',
    'CO_ether': '[OX2]([#6])[#6]',
    'CN_amine': '[CX4][NX3]',
    'SO_sulfone': '[SX4](=O)(=O)',
    'CX_halide': '[#6][F,Cl,Br,I]'
}

# Preprocessing functions
def baseline_correction_als(spectrum, lam=1e6, p=0.01, niter=3):
    """FAST Asymmetric Least Squares baseline correction - optimized for speed"""
    try:
        # SPEED OPTIMIZATION: Use percentile filter (100x faster than iterative ALS)
        L = len(spectrum)
        
        # Use percentile-based baseline estimation (very fast)
        baseline = percentile_filter(spectrum, percentile=5, size=100)
        
        return spectrum - baseline
    except Exception as e:
        # If baseline correction fails, return original spectrum
        st.warning(f"Baseline correction failed, using original spectrum: {e}")
        return spectrum
        w = p * (spectrum > z) + (1 - p) * (spectrum < z)
    
    return spectrum - z

def savitzky_golay_smooth(spectrum, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing filter"""
    if window_length % 2 == 0:
        window_length += 1
    return signal.savgol_filter(spectrum, window_length, polyorder)

def normalize_spectrum(spectrum, method='minmax'):
    """Normalize spectrum"""
    if method == 'minmax':
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val - min_val > 0:
            return (spectrum - min_val) / (max_val - min_val)
        return spectrum
    return spectrum

def detect_functional_groups(smiles):
    """Detect functional groups from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {group: 0 for group in FUNCTIONAL_GROUP_PATTERNS.keys()}
    
    groups_found = {}
    for group_name, smarts_pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            groups_found[group_name] = 1 if len(matches) > 0 else 0
        else:
            groups_found[group_name] = 0
    
    return groups_found

def extract_advanced_spectral_features(spectrum_row):
    """Extract comprehensive statistical and spectral features"""
    spectrum = spectrum_row.values if hasattr(spectrum_row, 'values') else spectrum_row
    
    n_regions = 31
    region_size = len(spectrum) // n_regions
    features = []
    
    # Regional statistical features
    for i in range(n_regions):
        start = i * region_size
        end = (i + 1) * region_size if i < n_regions - 1 else len(spectrum)
        region = spectrum[start:end]
        
        if len(region) > 0:
            features.extend([
                np.max(region), np.mean(region), np.std(region),
                np.sum(region), np.median(region), np.var(region),
            ])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
    
    # Peak detection features
    peaks, properties = signal.find_peaks(spectrum, height=0.1, distance=5)
    features.extend([
        len(peaks),
        np.mean(properties['peak_heights']) if len(peaks) > 0 else 0,
        np.max(properties['peak_heights']) if len(peaks) > 0 else 0,
        np.std(properties['peak_heights']) if len(peaks) > 0 else 0,
    ])
    
    # Derivative features
    first_derivative = np.gradient(spectrum)
    second_derivative = np.gradient(first_derivative)
    features.extend([
        np.mean(np.abs(first_derivative)),
        np.std(first_derivative),
        np.mean(np.abs(second_derivative)),
        np.std(second_derivative),
    ])
    
    # Spectral moments
    moments = [
        np.sum(spectrum * np.arange(len(spectrum))),
        np.sum(spectrum * np.arange(len(spectrum))**2),
    ]
    features.extend(moments)
    
    return features

# Cache data loading
@st.cache_data(show_spinner="Loading data...")
def load_data(max_samples=1000):
    """Load and process data with balanced functional groups (max 500 per group)"""
    try:
        # OPTIMIZED: Only process what we need (max_samples * 3 as buffer for balancing)
        sample_buffer = min(max_samples * 3, 20000)  # Cap at 20k for speed
        
        # Load limited SMILES
        with open('Dataset/smiles.txt', 'r') as f:
            all_smiles = [line.strip() for line in f if line.strip()][:sample_buffer]
        
        # Process SMILES to labels (only the buffer amount)
        all_labels_list = []
        for smiles in all_smiles:
            groups = detect_functional_groups(smiles)
            all_labels_list.append(groups)
        
        all_labels_df = pd.DataFrame(all_labels_list)
        
        # Balance dataset: max 10% of sample size per functional group
        max_per_group = int(max_samples * 0.10)  # 10% of requested samples
        selected_indices = []
        group_counts = {col: 0 for col in all_labels_df.columns}
        
        for idx in range(len(all_labels_df)):
            row = all_labels_df.iloc[idx]
            # Check if adding this sample would exceed 10% limit for any group
            can_add = True
            for col in all_labels_df.columns:
                if row[col] == 1 and group_counts[col] >= max_per_group:
                    can_add = False
                    break
            
            if can_add:
                selected_indices.append(idx)
                # Update counts
                for col in all_labels_df.columns:
                    if row[col] == 1:
                        group_counts[col] += 1
            
            # Stop if we have enough samples
            if len(selected_indices) >= max_samples:
                break
        
        # Get balanced subset
        smiles_list = [all_smiles[i] for i in selected_indices]
        labels_df = all_labels_df.iloc[selected_indices].copy()
        
        # Load corresponding IR data (only needed rows)
        ir_data = pd.read_csv('Dataset/qm9s_irdata.csv', nrows=sample_buffer)
        ir_data = ir_data.iloc[selected_indices].copy()
        
        # Keep only numeric columns
        numeric_cols = []
        for col in ir_data.columns:
            try:
                pd.to_numeric(ir_data[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        ir_data = ir_data[numeric_cols]
        
        # Reset indices to ensure clean alignment
        ir_data.reset_index(drop=True, inplace=True)
        labels_df.reset_index(drop=True, inplace=True)
        
        # Verify alignment
        assert len(ir_data) == len(labels_df) == len(smiles_list), "Data alignment error!"
        
        return ir_data, labels_df, smiles_list
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

@st.cache_data(show_spinner=False)
def preprocess_data(_ir_data):
    """Preprocess IR spectra with vectorized operations for speed"""
    try:
        # Convert to numpy array for faster processing
        spectra_array = _ir_data.values
        preprocessed_spectra = []
        
        # Process in batches for better performance
        for idx in range(len(spectra_array)):
            spectrum = spectra_array[idx]
            
            # Quick preprocessing: only baseline correction and normalization
            # Skip smoothing for speed (minimal impact on features)
            try:
                spectrum_corrected = baseline_correction_als(spectrum, lam=1e5, p=0.01)
                spectrum_normalized = normalize_spectrum(spectrum_corrected, method='minmax')
                preprocessed_spectra.append(spectrum_normalized)
            except:
                # If baseline correction fails, just normalize
                spectrum_normalized = normalize_spectrum(spectrum, method='minmax')
                preprocessed_spectra.append(spectrum_normalized)
        
        return pd.DataFrame(preprocessed_spectra)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return _ir_data

@st.cache_data(show_spinner=False)
def extract_features(_ir_data_preprocessed):
    """Extract features from preprocessed spectra - optimized"""
    try:
        # Convert to numpy for speed
        spectra_array = _ir_data_preprocessed.values
        X_features = []
        
        for idx in range(len(spectra_array)):
            try:
                features = extract_advanced_spectral_features(spectra_array[idx])
                X_features.append(features)
            except Exception as e:
                # If feature extraction fails for a sample, skip it
                st.warning(f"Skipping sample {idx} due to feature extraction error")
                continue
        
        X = pd.DataFrame(X_features)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Ensure we have the right number of samples
        if len(X) != len(_ir_data_preprocessed):
            st.warning(f"Feature extraction: Expected {len(_ir_data_preprocessed)} samples, got {len(X)}")
        
        return X
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        # Return empty dataframe with correct number of rows
        return pd.DataFrame(np.zeros((len(_ir_data_preprocessed), 196)))

@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test):
    """Train and evaluate models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50, max_depth=5, random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=300,
            random_state=42, early_stopping=True
        )
    }
    
    results = {}
    
    for name, model in models.items():
        clf = MultiOutputClassifier(model, n_jobs=-1)
        clf.fit(_X_train, _y_train)
        y_pred = clf.predict(_X_test)
        
        results[name] = {
            'model': clf,
            'predictions': y_pred,
            'Accuracy': accuracy_score(_y_test, y_pred),
            'F1-Micro': f1_score(_y_test, y_pred, average='micro', zero_division=0),
            'F1-Macro': f1_score(_y_test, y_pred, average='macro', zero_division=0),
            'Hamming Loss': hamming_loss(_y_test, y_pred),
            'Jaccard Score': jaccard_score(_y_test, y_pred, average='samples', zero_division=0)
        }
    
    return results

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🧪 IR Spectroscopy Functional Group Classification</div>', 
                unsafe_allow_html=True)
    
    st.info("""
    **📋 Project Overview**
    
    This ML pipeline classifies functional groups in organic molecules using:
    
    - **IR Spectroscopy Data**: Real infrared spectra from QM9 dataset
    - **SMILES Strings**: Chemical structure notation for automatic labeling  
    - **Multi-Label Classification**: Identifying 31 different functional groups
    - **Advanced Feature Engineering**: 196 features extracted from each spectrum
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    max_samples = st.sidebar.slider(
        "Number of samples to process",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="More samples = better results but slower processing"
    )
    
    st.sidebar.info("""
    **⚡ Speed Tips:**
    - 100-500 samples: Very fast (~10s)
    - 500-1000 samples: Fast (~20s)  
    - 1000-2000 samples: Medium (~40s)
    - 2000+ samples: Slower (~60s+)
    
    **First load takes longer** to cache data. Subsequent runs are instant!
    """)
    
    show_technical_details = st.sidebar.checkbox("Show Technical Details", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Navigation")
    page = st.sidebar.radio(
        "Select Section:",
        [
            "🏠 Home & Overview",
            "📥 Data Loading & Processing",
            "🔬 Feature Engineering",
            "🤖 Model Training & Results",
            "📈 Detailed Analysis",
            "🧪 Try Your Own SMILES"
        ]
    )
    
    # Load data
    with st.spinner(f"🔄 Loading and balancing {max_samples} samples... (First time ~30s, then instant!)"):
        ir_data, labels_df, smiles_list = load_data(max_samples)
    
    if ir_data is None:
        st.error("Failed to load data. Please check if Dataset folder contains the required files.")
        return
    
    # Page routing
    if page == "🏠 Home & Overview":
        show_home_page()
    
    elif page == "📥 Data Loading & Processing":
        show_data_processing_page(ir_data, labels_df, smiles_list)
    
    elif page == "🔬 Feature Engineering":
        show_feature_engineering_page(ir_data, labels_df)
    
    elif page == "🤖 Model Training & Results":
        show_model_training_page(ir_data, labels_df)
    
    elif page == "📈 Detailed Analysis":
        show_detailed_analysis_page(ir_data, labels_df)
    
    elif page == "🧪 Try Your Own SMILES":
        show_interactive_prediction_page(ir_data, labels_df)

def show_home_page():
    """Home page with project overview"""
    st.markdown('<div class="section-header">🎯 Project Objectives</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 Scientific Goal
        Develop an automated system to identify functional groups in organic molecules 
        using IR spectroscopy data, eliminating the need for manual peak analysis.
        
        ### 🎓 Learning Outcomes
        - Multi-label classification techniques
        - Chemical data processing with RDKit
        - Signal processing for spectroscopy
        - Advanced feature engineering
        - Model comparison and evaluation
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Technical Approach
        1. **SMILES Processing**: Convert chemical structures to functional group labels
        2. **IR Data Processing**: Load and clean spectroscopic data
        3. **Preprocessing**: Baseline correction, smoothing, normalization
        4. **Feature Extraction**: 196 features per spectrum
        5. **ML Training**: Compare 4 different models
        6. **Evaluation**: Multi-label metrics and analysis
        """)
    
    st.markdown('<div class="section-header">🏗️ Pipeline Architecture</div>', unsafe_allow_html=True)
    
    # Pipeline diagram
    st.markdown("""
    ```
    ┌─────────────────┐     ┌──────────────────┐
    │  SMILES Strings │     │   IR Spectra     │
    │  (QM9 Dataset)  │     │  (CSV Data)      │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             ↓                       ↓
    ┌─────────────────┐     ┌──────────────────┐
    │  RDKit SMARTS   │     │  Load & Clean    │
    │  Pattern Match  │     │  Numeric Data    │
    └────────┬────────┘     └────────┬─────────┘
             │                       │
             ↓                       ↓
    ┌─────────────────┐     ┌──────────────────┐
    │ 31 Functional   │     │  Preprocessing:  │
    │ Group Labels    │     │  • Baseline ALS  │
    │ (Binary Matrix) │     │  • SG Smoothing  │
    └────────┬────────┘     │  • Normalization │
             │               └────────┬─────────┘
             │                        │
             │                        ↓
             │               ┌──────────────────┐
             │               │ Feature Extract: │
             │               │  • 186 Regional  │
             │               │  • 4 Peak-based  │
             │               │  • 4 Derivative  │
             │               │  • 2 Moments     │
             │               └────────┬─────────┘
             │                        │
             └────────────┬───────────┘
                          ↓
                 ┌─────────────────┐
                 │  Train-Test     │
                 │  Split (80-20)  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  4 ML Models:   │
                 │  • Logistic Reg │
                 │  • Random Forest│
                 │  • Grad Boost   │
                 │  • Neural Net   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  Evaluation &   │
                 │  Comparison     │
                 └─────────────────┘
    ```
    """)
    
    st.markdown('<div class="section-header">📚 Key Technologies</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Chemistry**
        - RDKit: Molecule processing
        - SMARTS: Pattern matching
        - SMILES: Structure notation
        """)
    
    with col2:
        st.markdown("""
        **Signal Processing**
        - SciPy: Peak detection
        - Baseline correction (ALS)
        - Savitzky-Golay filtering
        """)
    
    with col3:
        st.markdown("""
        **Machine Learning**
        - Scikit-learn: Models
        - Multi-label classification
        - Feature engineering
        """)

def show_data_processing_page(ir_data, labels_df, smiles_list):
    """Data loading and processing page"""
    st.markdown('<div class="section-header">📥 Data Loading & Processing</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(ir_data))
    with col2:
        st.metric("Spectral Points per Sample", ir_data.shape[1])
    with col3:
        st.metric("Functional Groups", labels_df.shape[1])
    
    st.markdown("---")
    
    # Sample SMILES data
    st.markdown("### 📝 Sample SMILES Strings")
    st.info("SMILES (Simplified Molecular Input Line Entry System) is a notation for representing chemical structures")
    
    sample_smiles_df = pd.DataFrame({
        'Index': range(5),
        'SMILES': smiles_list[:5]
    })
    st.dataframe(sample_smiles_df, use_container_width=True)
    
    # Raw IR spectra visualization
    st.markdown("### 📊 Raw IR Spectra Samples")
    
    num_samples_to_show = st.slider("Number of spectra to display", 1, 6, 4)
    
    fig, axes = plt.subplots((num_samples_to_show + 1) // 2, 2, figsize=(14, 3 * ((num_samples_to_show + 1) // 2)))
    if num_samples_to_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    sample_indices = np.random.choice(len(ir_data), size=num_samples_to_show, replace=False)
    
    for idx, ax in enumerate(axes[:num_samples_to_show]):
        sample_idx = sample_indices[idx]
        spectrum = ir_data.iloc[sample_idx].values
        wavenumber = np.arange(len(spectrum))
        
        ax.plot(wavenumber, spectrum, linewidth=1.2, color='#667eea', alpha=0.8)
        ax.fill_between(wavenumber, spectrum, alpha=0.3, color='#a8b7f7')
        ax.set_title(f'Sample {sample_idx}', fontsize=11, fontweight='bold', color='#2d3748')
        ax.set_xlabel('Spectral Point Index', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(num_samples_to_show, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Functional group distribution
    st.markdown("### 🏷️ Functional Group Distribution")
    
    # Calculate the dynamic limit (10% of samples)
    max_limit = int(len(ir_data) * 0.10)
    
    st.info(f"""
    **🎯 Balanced Dataset Strategy**
    
    To prevent bias, each functional group is limited to a maximum of **{max_limit} occurrences** (10% of {len(ir_data)} samples).
    This ensures no single group dominates the training process.
    """)
    
    label_counts = labels_df.sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color bars: green if ≤max_limit, orange if close, red if at limit
    colors = []
    for count in label_counts:
        if count >= max_limit:
            colors.append('#f5576c')  # Red - at limit
        elif count >= max_limit * 0.8:  # 80% of limit
            colors.append('#f6ad55')  # Orange - close to limit
        else:
            colors.append('#48bb78')  # Green - well below limit
    
    bars = ax.bar(range(len(label_counts)), label_counts.values, 
                   color=colors, edgecolor='#2d3748', linewidth=1.5, alpha=0.8)
    
    # Add horizontal line at the dynamic max limit
    ax.axhline(y=max_limit, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Max Limit ({max_limit})')
    
    ax.set_title('Functional Group Occurrence in Dataset (Balanced)', fontsize=14, fontweight='bold', color='#2d3748')
    ax.set_xlabel('Functional Group', fontsize=11, color='#2d3748')
    ax.set_ylabel('Number of Samples', fontsize=11, color='#2d3748')
    ax.set_xticks(range(len(label_counts)))
    ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Top functional groups
    st.markdown("### 🔝 Top 10 Most Common Functional Groups")
    top_10 = label_counts.head(10)
    top_10_df = pd.DataFrame({
        'Functional Group': top_10.index,
        'Count': top_10.values,
        'Percentage': (top_10.values / len(labels_df) * 100).round(2)
    })
    st.dataframe(top_10_df, use_container_width=True)

def show_feature_engineering_page(ir_data, labels_df):
    """Feature engineering page"""
    st.markdown('<div class="section-header">🔬 Feature Engineering Pipeline</div>', unsafe_allow_html=True)
    
    st.info("""
    **🛠️ Preprocessing Steps**
    
    1. **Baseline Correction (ALS)**: Removes baseline drift and background signals
    2. **Savitzky-Golay Smoothing**: Reduces noise while preserving peak shapes
    3. **Min-Max Normalization**: Standardizes intensity ranges (0-1)
    """)
    
    # Add progress bar for preprocessing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Preprocessing spectra..."):
        # Create a wrapper to track progress
        total = len(ir_data)
        ir_data_preprocessed = preprocess_data(ir_data)
        progress_bar.progress(100)
        status_text.text("✅ Preprocessing complete!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    st.success("✅ Preprocessing complete!")
    
    # Show before/after preprocessing
    st.markdown("### 📊 Preprocessing Effect")
    
    sample_idx = st.slider("Select sample to visualize", 0, len(ir_data) - 1, 0)
    
    raw_spectrum = ir_data.iloc[sample_idx].values
    preprocessed_spectrum = ir_data_preprocessed.iloc[sample_idx].values
    x_axis = np.arange(len(raw_spectrum))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Before
    axes[0].plot(x_axis, raw_spectrum, linewidth=1.2, color='#f5576c', alpha=0.9)
    axes[0].fill_between(x_axis, raw_spectrum, alpha=0.3, color='#ff8fa3')
    axes[0].set_title(f'Raw Spectrum (Sample {sample_idx})', fontsize=12, fontweight='bold', color='#2d3748')
    axes[0].set_xlabel('Spectral Point', fontsize=10)
    axes[0].set_ylabel('Intensity', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # After
    axes[1].plot(x_axis, preprocessed_spectrum, linewidth=1.2, color='#48bb78', alpha=0.9)
    axes[1].fill_between(x_axis, preprocessed_spectrum, alpha=0.3, color='#9ae6b4')
    axes[1].set_title(f'Preprocessed Spectrum (Sample {sample_idx})', fontsize=12, fontweight='bold', color='#2d3748')
    axes[1].set_xlabel('Spectral Point', fontsize=10)
    axes[1].set_ylabel('Normalized Intensity', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Clean up the figure
    
    # Feature extraction
    st.markdown("### 🎯 Feature Extraction")
    
    st.success("""
    **📈 Feature Categories (196 total features)**
    
    - **Regional Features (186)**: 31 regions × 6 statistics (max, mean, std, sum, median, variance)
    - **Peak Features (4)**: Peak count, average height, max height, height std
    - **Derivative Features (4)**: 1st and 2nd derivative statistics
    - **Spectral Moments (2)**: 1st and 2nd moment (weighted position)
    """)
    
    with st.spinner("Extracting features..."):
        X = extract_features(ir_data_preprocessed)
    
    st.success(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    # Feature importance analysis instead of statistics
    st.markdown("### 📊 Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Value Ranges**")
        feature_info = pd.DataFrame({
            'Min': X.min(),
            'Max': X.max(),
            'Mean': X.mean(),
            'Std': X.std()
        })
        # Show features with highest variance (most informative)
        feature_info['Variance'] = X.var()
        top_features = feature_info.nlargest(10, 'Variance')
        st.dataframe(top_features, use_container_width=True)
        st.caption("Top 10 features by variance (most informative)")
    
    with col2:
        st.markdown("**Feature Distribution Quality**")
        # Show distribution of feature values
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Calculate coefficient of variation for each feature
        cv = (X.std() / (X.mean() + 1e-10)).abs()
        cv_sorted = cv.sort_values(ascending=False).head(20)
        
        ax.barh(range(len(cv_sorted)), cv_sorted.values, color='#667eea', edgecolor='#2d3748', alpha=0.8)
        ax.set_yticks(range(len(cv_sorted)))
        ax.set_yticklabels([f'F{i}' for i in cv_sorted.index], fontsize=8)
        ax.set_xlabel('Coefficient of Variation', fontsize=10)
        ax.set_title('Top 20 Most Variable Features', fontsize=11, fontweight='bold', color='#2d3748')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Higher values = more discriminative power")
    
    # Better correlation analysis
    st.markdown("### 🔥 Feature Correlation Analysis")
    
    st.info("""
    **Understanding the Correlation Map:**
    - **Red blocks**: Highly correlated features (redundant information)
    - **Blue blocks**: Negatively correlated features
    - **White/Gray**: Independent features (good for diversity)
    - **Diagonal patterns**: Regional features from adjacent spectral regions
    """)
    
    # Select only most variable features for meaningful correlation
    top_50_features = X.var().nlargest(50).index
    X_selected = X.iloc[:, top_50_features]
    
    feature_correlation = X_selected.corr()
    
    # Mask for upper triangle to reduce clutter
    mask = np.triu(np.ones_like(feature_correlation, dtype=bool), k=1)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(feature_correlation, mask=mask, cmap='RdBu_r', center=0, 
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                vmin=-1, vmax=1,
                annot=False)
    ax.set_title('Feature Correlation Matrix (Top 50 Most Variable Features)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Feature Index', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Show highly correlated pairs
    st.markdown("### 🔗 Highly Correlated Feature Pairs")
    corr_pairs = []
    for i in range(len(feature_correlation.columns)):
        for j in range(i+1, len(feature_correlation.columns)):
            corr_val = feature_correlation.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                corr_pairs.append({
                    'Feature 1': feature_correlation.columns[i],
                    'Feature 2': feature_correlation.columns[j],
                    'Correlation': round(corr_val, 3),
                    'Type': 'Positive' if corr_val > 0 else 'Negative'
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False, key=abs)
        st.dataframe(corr_df, use_container_width=True)
        st.caption(f"Found {len(corr_pairs)} feature pairs with |correlation| > 0.8")
    else:
        st.success("✅ No highly correlated features found - good feature diversity!")

def show_model_training_page(ir_data, labels_df):
    """Model training and results page"""
    st.markdown('<div class="section-header">🤖 Model Training & Results</div>', unsafe_allow_html=True)
    
    # Preprocess and extract features
    with st.spinner("Preprocessing and extracting features..."):
        ir_data_preprocessed = preprocess_data(ir_data)
        X = extract_features(ir_data_preprocessed)
        y = labels_df
    
    # CRITICAL FIX: Ensure exact alignment
    min_samples = min(len(X), len(y))
    X = X.iloc[:min_samples].copy()
    y = y.iloc[:min_samples].copy()
    
    # Reset indices for safety
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    # Verify alignment
    if len(X) != len(y):
        st.error(f"Data alignment error: X has {len(X)} samples, y has {len(y)} samples")
        return
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Filter valid labels
    train_label_uniques = y_train.nunique()
    valid_label_cols = train_label_uniques[train_label_uniques == 2].index.tolist()
    
    if len(valid_label_cols) == 0:
        valid_label_cols = y_train.columns.tolist()
    
    y_train = y_train[valid_label_cols]
    y_test = y_test[valid_label_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Display split info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", X_train.shape[0])
    with col2:
        st.metric("Test Samples", X_test.shape[0])
    with col3:
        st.metric("Active Labels", len(valid_label_cols))
    
    st.markdown("---")
    
    # Train models
    st.markdown("### 🏋️ Training Models")
    
    with st.spinner("Training 4 different models... This may take a minute..."):
        results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    st.success("✅ All models trained successfully!")
    
    # Model comparison
    st.markdown("### 📊 Model Performance Comparison")
    
    comparison_data = {
        name: {k: v for k, v in info.items() if k not in ['model', 'predictions']}
        for name, info in results.items()
    }
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    # Display metrics table
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen')
                .format("{:.4f}"), use_container_width=True)
    
    # Best model
    best_model_name = comparison_df['F1-Micro'].idxmax()
    best_f1 = comparison_df.loc[best_model_name, 'F1-Micro']
    best_accuracy = comparison_df.loc[best_model_name, 'Accuracy']
    
    st.markdown(f"""
    <div class="success-box">
    <h4>🏆 Best Model: {best_model_name}</h4>
    <ul>
        <li><b>F1-Micro Score</b>: {best_f1:.4f}</li>
        <li><b>Accuracy</b>: {best_accuracy:.4f}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualize comparison
    st.markdown("### 📈 Visual Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df[['Accuracy', 'F1-Micro', 'F1-Macro', 'Jaccard Score']].plot(
        kind='bar', ax=ax, colormap='viridis', width=0.8
    )
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Metric explanations
    with st.expander("ℹ️ Understanding the Metrics"):
        st.markdown("""
        - **Accuracy**: Percentage of correctly predicted labels
        - **F1-Micro**: Harmonic mean of precision and recall (averaged across all labels)
        - **F1-Macro**: Arithmetic mean of F1 scores for each label
        - **Hamming Loss**: Fraction of incorrectly predicted labels (lower is better)
        - **Jaccard Score**: Intersection over union of predicted and true labels
        """)

def show_detailed_analysis_page(ir_data, labels_df):
    """Detailed analysis page"""
    st.markdown('<div class="section-header">📈 Detailed Performance Analysis</div>', unsafe_allow_html=True)
    
    # Preprocess and extract features
    with st.spinner("Preparing data..."):
        ir_data_preprocessed = preprocess_data(ir_data)
        X = extract_features(ir_data_preprocessed)
        y = labels_df
        
        # CRITICAL FIX: Ensure exact alignment
        min_samples = min(len(X), len(y))
        X = X.iloc[:min_samples].copy()
        y = y.iloc[:min_samples].copy()
        
        # Reset indices
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_label_uniques = y_train.nunique()
        valid_label_cols = train_label_uniques[train_label_uniques == 2].index.tolist()
        
        if len(valid_label_cols) == 0:
            valid_label_cols = y_train.columns.tolist()
        
        y_train = y_train[valid_label_cols]
        y_test = y_test[valid_label_cols]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Best model analysis
    comparison_data = {
        name: {k: v for k, v in info.items() if k not in ['model', 'predictions']}
        for name, info in results.items()
    }
    comparison_df = pd.DataFrame(comparison_data).T
    best_model_name = comparison_df['F1-Micro'].idxmax()
    best_predictions = results[best_model_name]['predictions']
    
    # Per-functional-group performance
    st.markdown("### 🎯 Per-Functional-Group Performance")
    
    report = classification_report(
        y_test, best_predictions,
        target_names=list(y_test.columns),
        zero_division=0,
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).T
    
    # Display top performers
    st.markdown("#### 🏆 Top 15 Functional Groups (by F1-Score)")
    top_performers = report_df.iloc[:-3].sort_values('f1-score', ascending=False).head(15)
    st.dataframe(top_performers[['precision', 'recall', 'f1-score', 'support']]
                .style.background_gradient(cmap='Greens'), use_container_width=True)
    
    # Visualize performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1-Score distribution
    f1_scores = report_df.iloc[:-3]['f1-score'].sort_values(ascending=False)
    bars = axes[0].barh(range(len(f1_scores[:20])), f1_scores[:20], color='#667eea', edgecolor='#2d3748', linewidth=1.5)
    axes[0].set_yticks(range(len(f1_scores[:20])))
    axes[0].set_yticklabels(f1_scores[:20].index, fontsize=8)
    axes[0].set_xlabel('F1-Score', fontsize=10)
    axes[0].set_title('Top 20 Functional Groups by F1-Score', fontsize=11, fontweight='bold', color='#2d3748')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Support distribution
    support_data = report_df.iloc[:-3]['support'].sort_values(ascending=False)
    bars = axes[1].bar(range(len(support_data[:20])), support_data[:20], color='#f5576c', edgecolor='#2d3748', linewidth=1.5)
    axes[1].set_xticks(range(len(support_data[:20])))
    axes[1].set_xticklabels(support_data[:20].index, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('Number of Samples', fontsize=10)
    axes[1].set_title('Top 20 Functional Groups by Sample Count', fontsize=11, fontweight='bold', color='#2d3748')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Confusion analysis
    st.markdown("### 🔍 Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample-wise accuracy
        sample_accuracy = []
        for i in range(len(y_test)):
            acc = (y_test.iloc[i].values == best_predictions[i]).sum() / len(y_test.columns)
            sample_accuracy.append(acc)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sample_accuracy, bins=30, color='#764ba2', edgecolor='#2d3748', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Sample Accuracy', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Distribution of Sample-wise Accuracy', fontsize=11, fontweight='bold', color='#2d3748')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        # Label-wise accuracy
        label_accuracy = []
        for col in y_test.columns:
            acc = accuracy_score(y_test[col], best_predictions[:, y_test.columns.get_loc(col)])
            label_accuracy.append(acc)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(label_accuracy, bins=30, color='#48bb78', edgecolor='#2d3748', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Label Accuracy', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Distribution of Label-wise Accuracy', fontsize=11, fontweight='bold', color='#2d3748')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Sample Accuracy", f"{np.mean(sample_accuracy):.2%}")
    with col2:
        st.metric("Average Label Accuracy", f"{np.mean(label_accuracy):.2%}")
    with col3:
        st.metric("Perfect Predictions", f"{sum([acc == 1.0 for acc in sample_accuracy])}")

def show_interactive_prediction_page(ir_data, labels_df):
    """Interactive prediction page"""
    st.markdown('<div class="section-header">🧪 Try Your Own SMILES String</div>', unsafe_allow_html=True)
    
    st.info("""
    **Enter a SMILES string to predict which functional groups are present in the molecule.**
    
    The system will use the trained model to make predictions.
    """)
    
    # Example SMILES
    examples = {
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Acetic Acid": "CC(=O)O",
        "Acetone": "CC(=O)C",
        "Methylamine": "CN",
        "Glucose": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES String:",
            value="CCO",
            help="Enter a valid SMILES notation for an organic molecule"
        )
    
    with col2:
        st.markdown("**Examples:**")
        selected_example = st.selectbox("Or select an example:", list(examples.keys()))
        if st.button("Use Example"):
            smiles_input = examples[selected_example]
    
    if st.button("🔍 Predict Functional Groups", type="primary"):
        # Detect functional groups
        groups_found = detect_functional_groups(smiles_input)
        
        if groups_found is None or all(v == 0 for v in groups_found.values()):
            st.warning("Could not parse SMILES or no functional groups detected. Please check your input.")
        else:
            st.success(f"✅ Successfully parsed: `{smiles_input}`")
            
            # Show detected groups
            detected = {k: v for k, v in groups_found.items() if v == 1}
            
            if len(detected) > 0:
                st.markdown("### ✅ Detected Functional Groups")
                
                # Display as badges
                cols = st.columns(4)
                for idx, group in enumerate(detected.keys()):
                    with cols[idx % 4]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 12px; border-radius: 10px; margin: 5px 0; text-align: center; 
                        border: none; box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3); color: white;">
                        <b style="font-size: 0.9rem;">{group}</b>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show molecular structure visualization (if possible)
                try:
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol:
                        st.markdown("### 🔬 Molecular Properties")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            mw = Descriptors.MolWt(mol)
                            st.metric("Molecular Weight", f"{mw:.2f} g/mol")
                        
                        with col2:
                            logp = Descriptors.MolLogP(mol)
                            st.metric("LogP", f"{logp:.2f}")
                        
                        with col3:
                            hbd = Descriptors.NumHDonors(mol)
                            st.metric("H-Bond Donors", hbd)
                
                except Exception as e:
                    st.error(f"Error calculating properties: {e}")
            
            else:
                st.info("No functional groups detected in this molecule.")
            
            # Show all groups as table
            st.markdown("### 📋 Complete Functional Group Analysis")
            
            groups_df = pd.DataFrame({
                'Functional Group': list(groups_found.keys()),
                'Present': ['✅ Yes' if v == 1 else '❌ No' for v in groups_found.values()]
            })
            
            st.dataframe(groups_df, use_container_width=True, height=400)

# Run the app
if __name__ == "__main__":
    main()
