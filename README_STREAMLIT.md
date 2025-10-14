# 🧪 IR Spectroscopy Functional Group Classification - Streamlit Dashboard

## Overview
This interactive Streamlit dashboard presents a complete Machine Learning pipeline for classifying functional groups in organic molecules using IR spectroscopy data and SMILES strings.

## Features

### 📊 Interactive Pages

1. **🏠 Home & Overview**
   - Project objectives and learning outcomes
   - Complete pipeline architecture diagram
   - Technology stack overview

2. **📥 Data Loading & Processing**
   - Dataset statistics and metrics
   - Sample SMILES strings visualization
   - Raw IR spectra plotting
   - Functional group distribution analysis

3. **🔬 Feature Engineering**
   - Preprocessing pipeline visualization (Baseline correction, smoothing, normalization)
   - Before/after comparison of spectra
   - Feature extraction details (196 features)
   - Feature correlation heatmaps
   - Feature variance analysis

4. **🤖 Model Training & Results**
   - Train 4 different ML models simultaneously
   - Performance comparison table
   - Best model identification
   - Visual metric comparison charts
   - Detailed metric explanations

5. **📈 Detailed Analysis**
   - Per-functional-group performance metrics
   - Top performing functional groups
   - Sample-wise and label-wise accuracy distributions
   - Confusion analysis
   - Summary statistics

6. **🧪 Try Your Own SMILES**
   - Interactive prediction interface
   - Pre-loaded examples (Ethanol, Benzene, Acetic Acid, etc.)
   - Real-time functional group detection
   - Molecular property calculations
   - Visual badges for detected groups

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install RDKit (if not already installed)
```bash
pip install rdkit
```

## Running the Dashboard

### Basic Usage
```bash
streamlit run streamlit_app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Custom Port
```bash
streamlit run streamlit_app.py --server.port 8080
```

### Custom Configuration
```bash
streamlit run streamlit_app.py --server.maxUploadSize 200
```

## Usage Guide

### For Teaching Assistants Presentation

1. **Start with Home Page** - Explain the project overview and pipeline architecture
2. **Data Loading** - Show the raw data and preprocessing needs
3. **Feature Engineering** - Demonstrate the sophisticated preprocessing and feature extraction
4. **Model Training** - Live train models and compare performance (takes ~1-2 minutes)
5. **Detailed Analysis** - Deep dive into per-functional-group metrics
6. **Interactive Demo** - Let them try their own SMILES strings!

### Sidebar Controls

- **Number of samples**: Adjust from 100 to 5000
  - Start with 500-1000 for quick demos
  - Use 2000+ for more comprehensive results
  
- **Show Technical Details**: Toggle to show/hide advanced information

- **Navigation**: Jump between different sections easily

## Tips for Presentation

### 1. Quick Demo (5-10 minutes)
- Home page → Data Processing → Model Training → Interactive SMILES
- Use 500-1000 samples for faster processing

### 2. Comprehensive Demo (15-20 minutes)
- Cover all pages in order
- Use 1500-2000 samples
- Show feature engineering details
- Explain metrics in detail

### 3. Interactive Session
- Start on "Try Your Own SMILES" page
- Let TAs enter molecules they know
- Show how predictions work in real-time
- Examples to try:
  - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
  - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
  - Ethanol: `CCO`

## Key Features to Highlight

### Technical Excellence
✅ **31 Functional Groups** - Multi-label classification  
✅ **196 Features** - Advanced feature engineering  
✅ **4 ML Models** - Comprehensive comparison  
✅ **Real Data** - QM9 dataset with actual IR spectra  
✅ **Automated Labeling** - SMILES → SMARTS pattern matching  

### Processing Pipeline
✅ **Baseline Correction** - Asymmetric Least Squares (ALS)  
✅ **Noise Reduction** - Savitzky-Golay filtering  
✅ **Normalization** - Min-Max scaling  
✅ **Peak Detection** - SciPy signal processing  
✅ **Derivative Features** - Spectral shape analysis  

### Evaluation
✅ **Multiple Metrics** - Accuracy, F1, Hamming Loss, Jaccard  
✅ **Per-Group Analysis** - Individual functional group performance  
✅ **Visual Comparisons** - Interactive charts and graphs  

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'rdkit'`
- **Solution**: `pip install rdkit` or `conda install -c conda-forge rdkit`

**Issue**: Dashboard is slow with many samples
- **Solution**: Reduce sample count in sidebar (use 500-1000 for demos)

**Issue**: "Failed to load data" error
- **Solution**: Ensure `Dataset/qm9s_irdata.csv` and `Dataset/smiles.txt` exist

**Issue**: Models not training
- **Solution**: Increase sample count (need at least 100 samples)

## Dataset Requirements

The app expects the following files:
```
Dataset/
├── qm9s_irdata.csv    # IR spectroscopy data
└── smiles.txt         # SMILES strings (one per line)
```

## Performance Notes

- **Sample Size** vs **Processing Time**:
  - 500 samples: ~30 seconds
  - 1000 samples: ~1 minute
  - 2000 samples: ~2-3 minutes
  - 5000 samples: ~5-8 minutes

- First run will be slower due to caching
- Subsequent runs on same page are instant (cached)

## Customization

### Modify Sample Limit
Edit `streamlit_app.py` line ~213:
```python
max_samples = st.sidebar.slider(
    "Number of samples to process",
    min_value=100,
    max_value=5000,  # Change this
    value=1000,
    step=100
)
```

### Add More Models
Edit the `train_models()` function around line ~325 to add your own models.

### Change Visualization Colors
Modify the matplotlib/seaborn parameters in each plotting section.

## Contact & Support

For issues or questions about the dashboard:
1. Check the troubleshooting section above
2. Review the inline documentation in `streamlit_app.py`
3. Verify data files are in correct location

## License

This project is created for educational purposes as part of an ML course project.

---

**Pro Tip**: Use the "Try Your Own SMILES" page to make the presentation interactive and engaging! 🎓
