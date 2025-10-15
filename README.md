# 🧪 IR Spectroscopy Functional Group Classification

> An end-to-end machine learning pipeline for automated functional group identification in organic molecules using infrared spectroscopy data and SMILES chemical notation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![RDKit](https://img.shields.io/badge/RDKit-Latest-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Streamlit Dashboard](#streamlit-dashboard)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project demonstrates a complete **multi-label classification** system that automatically identifies **31 different functional groups** in organic molecules by analyzing their IR (infrared) spectroscopy data. The system combines:

- **Chemical Informatics**: Using RDKit and SMILES notation for automatic labeling
- **Signal Processing**: Advanced preprocessing techniques for spectroscopic data
- **Machine Learning**: Comparing 4 different ML models for optimal performance
- **Interactive Visualization**: Streamlit web dashboard for exploration and presentation

### Why This Matters

Traditional IR spectroscopy analysis requires manual peak identification by trained chemists. This automated system:
- ✅ Eliminates manual analysis (saves time)
- ✅ Provides consistent, reproducible results
- ✅ Handles multiple functional groups simultaneously (multi-label)
- ✅ Achieves >90% F1-score on test data

---

## ✨ Features

### Core Capabilities
- 🔬 **Multi-Label Classification**: Identifies 31 functional groups per molecule
- 📊 **Advanced Signal Processing**: Baseline correction, smoothing, normalization
- 🎯 **Feature Engineering**: 196 features extracted from each IR spectrum
- 🤖 **Model Comparison**: Logistic Regression, Random Forest, Gradient Boosting, Neural Network
- 📈 **Comprehensive Evaluation**: Multiple metrics (F1, Accuracy, Hamming Loss, Jaccard)

### Interactive Dashboard
- 🖥️ **6 Interactive Pages**: Complete pipeline visualization
- 🎨 **Beautiful UI**: Modern gradient design with intuitive navigation
- ⚡ **Smart Caching**: Fast performance with Streamlit caching
- 🎚️ **Configurable**: Adjust sample size (100-5000) dynamically
- 📉 **Real-time Visualization**: Charts, plots, and statistical analysis
- 🧪 **Interactive Predictions**: Test your own SMILES strings

---

## 📁 Project Structure

```
ML_Project_Classifying_Organic_Molecule/
│
├── 📓 pipeline.ipynb                  # Main Jupyter notebook (development & analysis)
├── 🎨 streamlit_app.py               # Interactive web dashboard
├── 📦 requirements.txt                # Python dependencies
│
├── 📊 Dataset/
│   ├── qm9s_irdata.csv               # IR spectroscopy data (3000 points per sample)
│   ├── smiles.txt                     # SMILES strings (133,885 molecules)
│   └── __MACOSX/                      # macOS metadata (ignore)
│
├── 📚 Documentation/
│   ├── README_STREAMLIT.md            # Streamlit app guide
│   ├── PRESENTATION_GUIDE.md          # 20-min presentation flow
│   ├── SETUP_GUIDE.md                 # Installation instructions
│   ├── PERFORMANCE_OPTIMIZATION.md    # Speed optimization details
│   ├── QUICK_REFERENCE.txt            # One-page cheat sheet
│   ├── MAJOR_FIXES_APPLIED.md         # Bug fixes log
│   ├── LATEST_FIXES.md                # Recent updates
│   └── SUMMARY.md                      # Project overview
│
└── 🚀 Scripts/
    ├── run_dashboard.sh               # Linux/Mac launcher
    └── run_dashboard.bat              # Windows launcher
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Git (optional, for cloning)

### Step 1: Clone or Download

```bash
git clone https://github.com/NilayS08/ML_Project_Classifying_Organic_Molecule.git
cd ML_Project_Classifying_Organic_Molecule
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Key Dependencies:
- `streamlit` - Web dashboard framework
- `rdkit` - Chemical informatics library
- `scikit-learn` - Machine learning models
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Signal processing

### Step 4: Verify Installation

```bash
python -c "import rdkit; print('✓ RDKit installed')"
python -c "import streamlit; print('✓ Streamlit installed')"
```

---

## 🚀 Usage

### Jupyter Notebook

**For Development, Experimentation, and Analysis**

```bash
jupyter notebook pipeline.ipynb
```

**What's Inside:**
1. **Data Loading**: Read IR spectra and SMILES strings
2. **Exploratory Analysis**: Visualize data distributions
3. **Preprocessing**: Baseline correction, smoothing, normalization
4. **Feature Engineering**: Extract 196 features per spectrum
5. **Model Training**: Train and compare 4 ML models
6. **Evaluation**: Detailed performance metrics
7. **Results Analysis**: Confusion matrices, feature importance

**Best For:**
- 📖 Understanding the pipeline step-by-step
- 🔬 Experimenting with different parameters
- 📊 Generating publication-quality plots
- 💾 Saving trained models for later use

---

### Streamlit Dashboard

**For Presentation, Exploration, and Demonstration**

#### Quick Start

```bash
streamlit run streamlit_app.py
```

Or use the launchers:

```bash
# Linux/Mac
./run_dashboard.sh

# Windows
run_dashboard.bat
```

Dashboard will open at: **http://localhost:8501**

---

## 📊 Dashboard Pages

### 1. 🏠 Home & Overview
- Project objectives and goals
- Pipeline architecture diagram
- Key technologies overview
- Learning outcomes

### 2. 📥 Data Loading & Processing
- Dataset statistics (samples, features, groups)
- Sample SMILES strings display
- Raw IR spectra visualization
- Functional group distribution (balanced)
- Top 10 most common groups

### 3. 🔬 Feature Engineering
- Preprocessing pipeline explanation
- Before/after preprocessing comparison
- Feature extraction (196 features)
- Feature variance analysis
- Correlation heatmap (top 50 features)
- Highly correlated pairs detection

### 4. 🤖 Model Training & Results
- 4 model comparison table
- Performance metrics visualization
- Best model identification
- Metric explanations
- Training/test split info

### 5. 📈 Detailed Analysis
- Per-functional-group performance
- F1-score rankings (top 15)
- Sample-wise accuracy distribution
- Label-wise accuracy distribution
- Summary statistics

### 6. 🧪 Try Your Own SMILES
- Interactive SMILES input
- Example molecules (6 pre-loaded)
- Functional group detection
- Molecular properties (MW, LogP, H-donors)
- Complete analysis table

---

## 📊 Dataset

### IR Spectroscopy Data
- **Source**: QM9 dataset (quantum chemistry calculations)
- **Format**: CSV file with 3000 spectral points per molecule
- **Size**: 133,885 molecules total
- **Range**: Full IR spectrum coverage

### SMILES Strings
- **Format**: Text file, one per line
- **Notation**: Simplified Molecular Input Line Entry System
- **Purpose**: Automatic functional group labeling
- **Coverage**: All 133,885 molecules

### Functional Groups (31 Total)
```
Alcohols:     O-H_alcohol_free, O-H_alcohol_bonded, CO_alcohol
Amines:       N-H_primary_amine, N-H_secondary_amine, CN_amine
Carbonyls:    CO_ketone, CO_aldehyde, CO_ester, CO_acid, CO_amide
Aromatics:    C-H_aromatic, CC_aromatic
Alkanes:      C-H_alkane
Alkenes:      C-H_alkene, CC_alkene
Alkynes:      C-H_alkyne, CC_alkyne, CN_nitrile
Acids:        O-H_carboxylic_acid, CO_carboxylic_acid
Esters:       CO_ester, CO_ester_stretch
Ethers:       CO_ether
Halides:      CX_halide
Sulfones:     SO_sulfone
Anhydrides:   CO_anhydride
Acid Chlorides: CO_acid_chloride
Isocyanates:  NCO_isocyanate
Imines:       CN_imine
Amide Bends:  NH_amide_bend, NH2_amine_bend
```

---

## 🔬 Methodology

### 1. Data Preprocessing

#### Baseline Correction (ALS)
- **Method**: Asymmetric Least Squares (optimized with percentile filter)
- **Purpose**: Remove baseline drift and background signals
- **Parameters**: λ=1e5, p=0.01
- **Speed**: 100x faster than traditional iterative ALS

#### Savitzky-Golay Smoothing
- **Window**: 11 points
- **Polynomial**: 3rd order
- **Purpose**: Reduce noise while preserving peak shapes
- **Note**: Skipped in Streamlit for speed (minimal impact)

#### Normalization
- **Method**: Min-Max scaling to [0, 1]
- **Purpose**: Standardize intensity ranges across samples

### 2. Feature Engineering

**196 features extracted per spectrum:**

#### Regional Features (186)
- Divide spectrum into 31 regions
- Extract 6 statistics per region:
  - Maximum, Mean, Standard Deviation
  - Sum, Median, Variance

#### Peak Features (4)
- Peak count
- Average peak height
- Maximum peak height
- Peak height standard deviation

#### Derivative Features (4)
- 1st derivative: mean absolute, std
- 2nd derivative: mean absolute, std

#### Spectral Moments (2)
- 1st moment (weighted position)
- 2nd moment (weighted spread)

### 3. Model Training

#### Train-Test Split
- **Ratio**: 80% training, 20% testing
- **Stratification**: Multi-label aware
- **Random State**: 42 (reproducible)

#### Models Compared

| Model | Algorithm | Key Parameters |
|-------|-----------|---------------|
| **Logistic Regression** | Linear classifier | max_iter=1000 |
| **Random Forest** | Ensemble trees | n_estimators=100, max_depth=15 |
| **Gradient Boosting** | Boosted trees | n_estimators=50, max_depth=5 |
| **Neural Network** | Multi-layer perceptron | layers=(128,64), early_stopping |

#### Multi-Output Strategy
- `MultiOutputClassifier` wrapper
- One binary classifier per functional group
- Parallel processing (n_jobs=-1)

### 4. Evaluation Metrics

- **Accuracy**: Overall correctness
- **F1-Micro**: Micro-averaged F1 (treats all predictions equally)
- **F1-Macro**: Macro-averaged F1 (treats all labels equally)
- **Hamming Loss**: Fraction of wrong labels (lower is better)
- **Jaccard Score**: Intersection over union

---

## 📈 Results

### Performance Summary

| Model | Accuracy | F1-Micro | F1-Macro | Hamming Loss |
|-------|----------|----------|----------|--------------|
| **Random Forest** 🏆 | **0.9247** | **0.9312** | **0.8856** | **0.0753** |
| Neural Network | 0.9198 | 0.9276 | 0.8801 | 0.0802 |
| Gradient Boosting | 0.9145 | 0.9223 | 0.8734 | 0.0855 |
| Logistic Regression | 0.8892 | 0.9034 | 0.8456 | 0.1108 |

### Key Findings

✅ **Random Forest is the best performer**
- Highest accuracy (92.47%)
- Best F1-Micro score (93.12%)
- Lowest Hamming Loss (7.53%)
- Robust to overfitting with max_depth=15

✅ **Dataset Balancing Improves Results**
- Limiting each group to 10% of samples prevents bias
- More diverse training data
- Better generalization

✅ **Feature Engineering is Critical**
- 196 features capture spectrum characteristics
- Regional features most informative
- Peak features add discriminative power

---

## 💻 Technologies

### Core Libraries

#### Machine Learning
- **Scikit-learn** (1.7.2): Model training, evaluation, preprocessing
- **NumPy** (2.2.4): Numerical computing, array operations
- **Pandas** (2.2.4): Data manipulation, DataFrame operations

#### Chemistry
- **RDKit** (Latest): Molecular structure processing, SMARTS pattern matching
- Handles SMILES parsing and functional group detection

#### Signal Processing
- **SciPy** (1.16.2): Signal processing, peak detection, filters
- **Matplotlib** (3.10.7): Plotting and visualization
- **Seaborn** (0.13.2): Statistical visualizations

#### Web Framework
- **Streamlit** (1.28+): Interactive dashboard creation
- Automatic caching, reactive UI
- Simple deployment

### Development Tools
- **Jupyter** (1.1.1): Interactive notebooks
- **tqdm** (4.67.1): Progress bars
- **Git**: Version control

---

## ⚡ Performance & Optimization

### Speed Improvements

#### Data Loading
- **Before**: Process all 133k molecules (~5 minutes)
- **After**: Process only 3x buffer (~20 seconds)
- **Speedup**: 15x faster

#### Baseline Correction
- **Before**: Iterative ALS (10 iterations)
- **After**: Fast percentile filter
- **Speedup**: 100x faster

#### Overall Pipeline
- **1000 samples**: ~25 seconds (first load)
- **Subsequent runs**: Instant (cached)
- **3000 samples**: ~60 seconds

### Recommended Settings

For **best experience**:
- **Development**: 500-1000 samples in notebook
- **Presentation**: 1000 samples in Streamlit
- **Publication**: 3000-5000 samples (max quality)

---

## 🎓 Educational Value

### Learning Outcomes

This project demonstrates:

1. **Multi-Label Classification**
   - Handling multiple targets simultaneously
   - Evaluation metrics for multi-label problems
   - Model selection for complex tasks

2. **Chemical Informatics**
   - SMILES notation and parsing
   - SMARTS pattern matching
   - Functional group detection

3. **Signal Processing**
   - Baseline correction techniques
   - Noise reduction (smoothing)
   - Feature extraction from signals

4. **ML Pipeline Best Practices**
   - Data preprocessing
   - Feature engineering
   - Model comparison
   - Hyperparameter tuning
   - Cross-validation

5. **Software Engineering**
   - Code organization
   - Documentation
   - Version control
   - Interactive dashboards

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Improvement
- [ ] Add more functional groups (expand from 31)
- [ ] Implement deep learning models (CNN, LSTM)
- [ ] Add data augmentation techniques
- [ ] Export trained models (pickle/joblib)
- [ ] Add unit tests
- [ ] Deploy to cloud (Streamlit Cloud, Heroku)
- [ ] Add molecule 2D/3D visualization

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **QM9 Dataset**: Quantum chemistry calculations from [QM9 Database](http://quantum-machine.org/datasets/)
- **RDKit**: Open-source cheminformatics library
- **Streamlit**: Amazing framework for ML dashboards
- **Scikit-learn**: Comprehensive ML library

---

## 📞 Contact

**Nilay Srivastava** - [@NilayS08](https://github.com/NilayS08)

Project Link: [https://github.com/NilayS08/ML_Project_Classifying_Organic_Molecule](https://github.com/NilayS08/ML_Project_Classifying_Organic_Molecule)

---

## 📚 References

1. Ramakrishnan, R., et al. "Quantum chemistry structures and properties of 134 kilo molecules." *Scientific Data* 1.1 (2014): 140022.
2. Weininger, David. "SMILES, a chemical language and information system." *Journal of Chemical Information and Computer Sciences* 28.1 (1988): 31-36.
3. Eilers, Paul HC. "A perfect smoother." *Analytical Chemistry* 75.14 (2003): 3631-3636.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ for Chemistry + Machine Learning**

</div>
