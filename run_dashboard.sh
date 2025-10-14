#!/bin/bash

# Quick Start Script for IR Spectroscopy Streamlit Dashboard
# This script sets up and runs the Streamlit application

echo "=============================================="
echo "IR Spectroscopy ML Project - Dashboard Launcher"
echo "=============================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit is not installed."
    echo "Installing required packages..."
    pip install -r requirements.txt
    echo "✅ Installation complete!"
    echo ""
fi

# Check if Dataset folder exists
if [ ! -d "Dataset" ]; then
    echo "❌ Error: Dataset folder not found!"
    echo "Please ensure the Dataset folder with required files exists."
    exit 1
fi

# Check for required data files
if [ ! -f "Dataset/qm9s_irdata.csv" ]; then
    echo "❌ Error: Dataset/qm9s_irdata.csv not found!"
    exit 1
fi

if [ ! -f "Dataset/smiles.txt" ]; then
    echo "❌ Error: Dataset/smiles.txt not found!"
    exit 1
fi

echo "✅ All data files found!"
echo ""
echo "🚀 Starting Streamlit Dashboard..."
echo ""
echo "The dashboard will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "📌 Tips for presentation:"
echo "   - Start with 500-1000 samples for quick demo"
echo "   - Use sidebar to navigate between sections"
echo "   - Try the 'Your Own SMILES' page for interactive demo"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=============================================="
echo ""

# Run streamlit
streamlit run streamlit_app.py
