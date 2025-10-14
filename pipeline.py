"""
IR Spectroscopy Functional Group Classification Pipeline
Uses SMILES strings to automatically label functional groups
Processes real IR spectra data for multi-label classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from tqdm import tqdm

# ML Libraries
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    hamming_loss, jaccard_score
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

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
plt.rcParams['figure.figsize'] = (14, 6)

print("=" * 80)
print("IR SPECTROSCOPY FUNCTIONAL GROUP CLASSIFICATION")
print("Real Data Pipeline with SMILES-based Labeling")
print("=" * 80)

# ============================================================================
# PART 1: FUNCTIONAL GROUP DETECTION FROM SMILES
# ============================================================================

# Define 31 functional groups with SMARTS patterns
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

def detect_functional_groups(smiles):
    """
    Detect functional groups from SMILES string using SMARTS patterns
    
    Parameters:
    -----------
    smiles : str
        SMILES string of molecule
        
    Returns:
    --------
    dict : Binary labels for each functional group
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # Invalid SMILES - return all zeros
        return {group: 0 for group in FUNCTIONAL_GROUP_PATTERNS.keys()}
    
    labels = {}
    for group_name, smarts_pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            labels[group_name] = 1 if len(matches) > 0 else 0
        else:
            labels[group_name] = 0
    
    return labels

print("\n[STEP 1] Loading SMILES and Generating Labels...")
print("-" * 80)

# Load SMILES file
SMILES_FILE = 'smiles.txt'
smiles_list = []

with open(SMILES_FILE, 'r') as f:
    for line in f:
        smiles_list.append(line.strip())

print(f"✓ Loaded {len(smiles_list)} SMILES strings")

# Generate functional group labels (process in batches for large datasets)
print("\nDetecting functional groups from SMILES...")
all_labels = []
batch_size = 1000

for i in tqdm(range(0, len(smiles_list), batch_size)):
    batch = smiles_list[i:i+batch_size]
    batch_labels = [detect_functional_groups(smi) for smi in batch]
    all_labels.extend(batch_labels)

# Create labels dataframe
labels_df = pd.DataFrame(all_labels)
print(f"\n✓ Generated labels for {len(labels_df)} compounds")
print(f"✓ Number of functional groups: {labels_df.shape[1]}")

# Show distribution
print("\nFunctional Group Distribution:")
fg_counts = labels_df.sum().sort_values(ascending=False)
print(fg_counts.head(15))

# ============================================================================
# PART 2: LOAD AND PREPROCESS IR SPECTRA DATA
# ============================================================================

print("\n[STEP 2] Loading IR Spectra Data...")
print("-" * 80)

IR_DATA_FILE = 'qm9s_irdata.csv'

# Load IR data in chunks (memory efficient for 4GB file)
print("Reading IR spectra (this may take a few minutes for large files)...")

chunks = []
chunk_size = 2000  # smaller chunks to reduce memory pressure

for chunk in pd.read_csv(IR_DATA_FILE, chunksize=chunk_size, header=None, low_memory=False):
    chunks.append(chunk)
    if len(chunks) % 20 == 0:
        print(f"  Processed {len(chunks) * chunk_size} spectra...")

ir_data = pd.concat(chunks, ignore_index=True)
print(f"\n✓ Loaded IR data: {ir_data.shape}")

# Identify and remove non-numeric columns (likely SMILES or index)
print("\nCleaning IR data (removing non-numeric columns)...")
numeric_cols = []
for col in ir_data.columns:
    try:
        # Try to convert column to float
        pd.to_numeric(ir_data[col], errors='raise')
        numeric_cols.append(col)
    except (ValueError, TypeError):
        # Skip non-numeric columns
        print(f"  Skipping non-numeric column: {col}")
        continue

# Keep only numeric columns
ir_data = ir_data[numeric_cols]
print(f"✓ Kept {len(numeric_cols)} numeric columns")
print(f"✓ Cleaned IR data shape: {ir_data.shape}")

# Ensure alignment
n_samples = min(len(smiles_list), len(ir_data))
ir_data = ir_data.iloc[:n_samples]
labels_df = labels_df.iloc[:n_samples]

print(f"\n✓ Final aligned dataset: {n_samples} samples")

# ============================================================================
# PART 3: FEATURE EXTRACTION FROM SPECTRA
# ============================================================================

print("\n[STEP 3] Extracting Features from IR Spectra...")
print("-" * 80)

def extract_spectral_features(spectrum_row):
    """
    Extract statistical features from IR spectrum
    
    Features per region:
    - Maximum intensity
    - Mean intensity
    - Standard deviation
    - Integrated area (sum)
    """
    # Convert to numeric, replacing any non-numeric values with NaN
    spectrum = pd.to_numeric(spectrum_row, errors='coerce').values
    
    # Remove NaN values
    spectrum = spectrum[~np.isnan(spectrum)]
    
    if len(spectrum) == 0:
        # Return zeros if spectrum is empty
        return [0] * (31 * 4)
    
    # Divide spectrum into 31 regions (one per functional group)
    n_regions = 31
    region_size = len(spectrum) // n_regions
    
    features = []
    for i in range(n_regions):
        start = i * region_size
        end = (i + 1) * region_size if i < n_regions - 1 else len(spectrum)
        region = spectrum[start:end]
        
        if len(region) > 0:
            features.extend([
                np.max(region),
                np.mean(region),
                np.std(region),
                np.sum(region)
            ])
        else:
            features.extend([0, 0, 0, 0])
    
    return features

# Extract features from all spectra
print("Extracting features (this may take several minutes)...")
X_features = []

for idx in tqdm(range(len(ir_data))):
    features = extract_spectral_features(ir_data.iloc[idx])
    X_features.append(features)

X = pd.DataFrame(X_features)
y = labels_df

print(f"\n✓ Feature matrix shape: {X.shape}")
print(f"✓ Label matrix shape: {y.shape}")

# ============================================================================
# PART 4: TRAIN-TEST SPLIT AND SCALING
# ============================================================================

print("\n[STEP 4] Preparing Data for Training...")
print("-" * 80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Identify usable labels: columns that contain at least two classes in the training set
train_label_uniques = y_train.nunique()
valid_label_cols = train_label_uniques[train_label_uniques == 2].index.tolist()
invalid_label_cols = [col for col in y_train.columns if col not in valid_label_cols]

if len(valid_label_cols) == 0:
    raise ValueError(
        "No label columns in the training set contain two classes. "
        "This can happen if all functional-group labels are all-zero or all-one. "
        "Please verify your SMILES labeling and IR/SMILES alignment."
    )

if len(invalid_label_cols) > 0:
    print("\nWarning: Skipping labels with a single class in training set (not learnable):")
    # Show a small subset to avoid flooding the console
    preview = invalid_label_cols[:10]
    print(f"  {len(invalid_label_cols)} labels skipped. Examples: {preview}{' ...' if len(invalid_label_cols) > 10 else ''}")

# Filter labels to only valid columns downstream
y_train = y_train[valid_label_cols]
y_test = y_test[valid_label_cols]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ============================================================================
# PART 5: TRAIN MACHINE LEARNING MODELS
# ============================================================================

print("\n[STEP 5] Training Machine Learning Models...")
print("=" * 80)

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
    print(f"\nTraining {name}...")
    
    # Wrap for multi-label classification
    clf = MultiOutputClassifier(model, n_jobs=-1)
    
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    results[name] = {
        'model': clf,
        'predictions': y_pred,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'F1-Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Hamming Loss': hamming_loss(y_test, y_pred),
        'Jaccard Score': jaccard_score(y_test, y_pred, average='samples', zero_division=0)
    }
    
    print(f"  ✓ Accuracy: {results[name]['Accuracy']:.4f}")
    print(f"  ✓ F1-Micro: {results[name]['F1-Micro']:.4f}")
    print(f"  ✓ F1-Macro: {results[name]['F1-Macro']:.4f}")

# ============================================================================
# PART 6: MODEL COMPARISON AND EVALUATION
# ============================================================================

print("\n[STEP 6] Model Performance Comparison...")
print("=" * 80)

comparison_data = {
    name: {k: v for k, v in info.items() if k not in ['model', 'predictions']}
    for name, info in results.items()
}

comparison_df = pd.DataFrame(comparison_data).T
print("\nModel Performance:")
print(comparison_df.to_string())

# Find best model
best_model_name = comparison_df['F1-Micro'].idxmax()
print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"F1-Micro: {comparison_df.loc[best_model_name, 'F1-Micro']:.4f}")
print(f"Accuracy: {comparison_df.loc[best_model_name, 'Accuracy']:.4f}")

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))
comparison_df[['Accuracy', 'F1-Micro', 'F1-Macro', 'Jaccard Score']].plot(
    kind='bar', ax=ax, colormap='viridis'
)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================================
# PART 7: DETAILED ANALYSIS
# ============================================================================

print("\n[STEP 7] Detailed Per-Functional-Group Analysis...")
print("-" * 80)

best_predictions = results[best_model_name]['predictions']

# Classification report
report = classification_report(
    y_test, best_predictions,
    target_names=list(y_test.columns),
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).T
print("\nTop 10 Functional Groups (by F1-Score):")
print(report_df.iloc[:-3][['precision', 'recall', 'f1-score', 'support']].head(10))

# Visualize functional group distribution
fig, ax = plt.subplots(figsize=(16, 6))
y_train.sum().sort_values(ascending=False).plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('Functional Group Distribution in Training Data', fontsize=14, fontweight='bold')
ax.set_xlabel('Functional Group')
ax.set_ylabel('Number of Occurrences')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================================================
# PART 8: SAVE RESULTS
# ============================================================================

print("\n[STEP 8] Saving Results...")
print("-" * 80)

# Save labels
labels_df.to_csv('functional_group_labels.csv', index=False)
print("✓ Saved: functional_group_labels.csv")

# Save model comparison
comparison_df.to_csv('model_comparison_results.csv')
print("✓ Saved: model_comparison_results.csv")

# Save detailed performance
report_df.to_csv('per_group_performance.csv')
print("✓ Saved: per_group_performance.csv")

# Save predictions
pred_df = pd.DataFrame(
    best_predictions, 
    columns=list(y_test.columns)
)
pred_df.to_csv('test_predictions.csv', index=False)
print("✓ Saved: test_predictions.csv")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print(f"✓ Processed {n_samples} compounds")
print(f"✓ Extracted {X.shape[1]} features per spectrum")
print(f"✓ Classified {y.shape[1]} functional groups")
print(f"✓ Best model: {best_model_name} (F1-Micro: {comparison_df.loc[best_model_name, 'F1-Micro']:.4f})")
print("=" * 80)