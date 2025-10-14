# Setup Guide for Streamlit Dashboard

## Quick Setup Instructions

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_app.py
```

### Option 2: Using System Python (if allowed)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_app.py
```

### Option 3: Using Bash Script (Linux/Mac)

```bash
# Make executable (first time only)
chmod +x run_dashboard.sh

# Run
./run_dashboard.sh
```

### Option 4: Using Batch File (Windows)

```cmd
# Double-click run_dashboard.bat
# OR run from command prompt:
run_dashboard.bat
```

## What Gets Installed

The dashboard requires:
- streamlit (web framework)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- scikit-learn (ML models)
- scipy (signal processing)
- rdkit (chemistry)
- Other dependencies (see requirements.txt)

## Running the Dashboard

Once installed, run:
```bash
streamlit run streamlit_app.py
```

The dashboard will open at: **http://localhost:8501**

## For Your Presentation

1. **Before the demo**: 
   - Test run with 500 samples to ensure everything works
   - Keep the browser window ready
   
2. **During the demo**:
   - Start with Home page to explain project
   - Navigate through each section systematically
   - Use "Try Your Own SMILES" for interactive demo
   
3. **Recommended flow**:
   ```
   Home → Data Processing → Feature Engineering → 
   Model Training → Detailed Analysis → Interactive Demo
   ```

## Troubleshooting

### Issue: Can't install packages
**Solution**: Use a virtual environment (see Option 1 above)

### Issue: RDKit installation fails
**Solution**: 
```bash
# Using conda (recommended)
conda install -c conda-forge rdkit

# OR using pip
pip install rdkit
```

### Issue: Dashboard won't start
**Solution**: Check that all dependencies are installed:
```bash
pip list | grep streamlit
pip list | grep rdkit
```

### Issue: "Failed to load data"
**Solution**: Verify files exist:
```bash
ls -la Dataset/
# Should show:
# - qm9s_irdata.csv
# - smiles.txt
```

## Performance Tips

- **For quick demos**: Use 500-1000 samples
- **For comprehensive results**: Use 2000-3000 samples
- **Maximum**: 5000 samples (takes ~5-8 minutes)

## Custom Configuration

You can customize Streamlit settings by creating `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
maxUploadSize = 200
```

## Need Help?

1. Check README_STREAMLIT.md for detailed documentation
2. Review inline comments in streamlit_app.py
3. Verify all data files are present in Dataset folder

---

**Ready to impress your TAs!** 🎓🚀
