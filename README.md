# 🔬 Molecular Flashpoint Prediction using Machine Learning


A comprehensive machine learning project that predicts molecular flashpoint values using chemical structure data (SMILES strings) and various molecular descriptors. This project demonstrates feature engineering techniques for cheminformatics and implements machine learning pipelines for molecular property prediction.

## 🎯 Project Overview

Flashpoint is a critical safety parameter in chemistry and chemical engineering, representing the lowest temperature at which a volatile liquid can vaporize to form an ignitable mixture with air. This project leverages machine learning to predict flashpoint values from molecular structure data, enabling rapid screening of chemical compounds for safety assessment.

### 🌟 Key Features

- ***Chemical Feature Engineering**: Convert SMILES strings to molecular descriptors using RDKit
- **Comprehensive Data Processing**: Handle categorical variables, binary features, and molecular fingerprints
- **Feature Importance Analysis**: Identify the most predictive molecular properties
- **Model Performance Evaluation**: Detailed analysis with multiple metrics and visualizations
- **Reproducible Pipeline**: Well-documented code with consistent random seeds

## 📊 Dataset

The dataset contains **14,696 molecular compounds** with the following features:

- **Target Variable**: `flashpoint` (temperature in Kelvin)
- **Molecular Structure**: `smiles` (SMILES notation strings)
- **Chemical Properties**: Binary indicators for silicon, metallic, tin, and acid compounds
- **Data Sources**: Multiple chemical databases (19 different sources)
- **Data Types**: Training, test, and validation splits




## 🔬 Methodology

### 1. Feature Engineering Pipeline

#### Chemical Structure Processing
- **SMILES → Molecular Descriptors**: Extract physicochemical properties
  - Molecular Weight (MolWt)
  - Lipophilicity (LogP)
  - Hydrogen Bond Donors/Acceptors
  - Topological Polar Surface Area (TPSA)
  - Rotatable Bonds Count
  - Aromatic Rings Count

#### Categorical Encoding
- **One-Hot Encoding**: For data types (4 categories)
- **Label Encoding**: For sources (19 categories)
- **Binary Features**: Already processed (0/1)

### 2. Machine Learning Model

**Random Forest Regressor** with optimized hyperparameters:
- 200 estimators for robust predictions
- Max depth of 15 to prevent overfitting
- Minimum samples split/leaf for regularization

### 3. Feature Selection Strategy

- **Initial Training**: Train Random Forest with all 15 engineered features
- **Feature Importance Analysis**: Rank features using Random Forest's built-in importance
- **Top Feature Selection**: Select top 10 most important features (33% reduction)
- **Model Optimization**: Retrain model using only selected features
- **Performance Validation**: Ensure minimal performance loss with reduced feature set

## 📁 Project Structure

```
supervised-learning-project/
│
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── data.csv              # Original dataset
│   │   └── meta.yaml             # Dataset metadata
│   └── 📂 processed/
│       ├── X_train.csv           # Processed training features
│       ├── X_test.csv            # Processed test features
│       ├── y_train.csv           # Training targets
│       ├── y_test.csv            # Test targets
│       └── feature_importance.csv # Feature importance rankings
│
├── 📂 models/
│   ├── rf_flashpoint_reduced_latest.pkl # Latest reduced model
│   ├── feature_info_latest.pkl          # Feature information
│   └── [timestamped model versions]     # Historical model versions
│
├── 📂 src/
│   ├── data_preprocessing.ipynb  # Data cleaning & feature engineering
│   ├── train_model.ipynb         # Model training & evaluation
│   ├── model_utils.py           # Model loading utilities
│   └── predict_demo.py          # Prediction examples
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Conda environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YassineBen-Yahia/Molecular-Flashpoint-Prediction
   cd "supervised learning project"
   ```

2. **Create conda environment**
   ```bash
   conda create -n chemml python=3.8
   conda activate chemml
   ```

3. **Install dependencies**
   ```bash
   # Install RDKit via conda (recommended)
   conda install -c conda-forge rdkit
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

### Running the Analysis

1. **Data Preprocessing** (`src/data_preprocessing.ipynb`)
   - Load and explore the dataset
   - Convert SMILES strings to molecular descriptors
   - Encode categorical variables
   - Generate train/test splits

2. **Model Training And Feature Selection** (`src/train_model.ipynb`)
   - Train Random Forest model with all features
   - Perform feature importance analysis
   - Select top 10 most important features
   - Train optimized reduced model
   - Evaluate model performance
   - Save reduced model to `models/` directory

3. **Using Saved Models** (`src/model_utils.py` & `src/predict_demo.py`)
   - Load pre-trained reduced model for predictions
   - Demo script showing prediction examples
   - Utilities for model management

## 🔮 Using the Trained Model

### Loading Saved Models

```python
from src.model_utils import load_latest_model, predict_flashpoint

# Load the trained reduced model
model = load_latest_model()

# Make predictions on new data
predictions = predict_flashpoint(model, new_data)
```

### Model Versions

The project saves multiple versions of the reduced model:
- **Latest version**: `rf_flashpoint_reduced_latest.pkl` for easy loading
- **Timestamped versions**: `rf_flashpoint_reduced_YYYYMMDD_HHMMSS.pkl` for version control
- **Feature information**: Complete metadata about selected features and performance

### Prediction Demo

Run the demo script to see the model in action:
```bash
cd src
python predict_demo.py
```

## 📈 Results

### Model Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **R² Score** | 0.9847 | 0.8923 |
| **RMSE** | 8.24 K | 21.87 K |
| **MAE** | 4.12 K | 12.45 K |

### Top Predictive Features

1. **TPSA** (Topological Polar Surface Area) - 15.2%
2. **MolWt** (Molecular Weight) - 12.8%
3. **NumAromaticRings** - 11.3%
4. **LogP** (Lipophilicity) - 9.7%
5. **NumRotatableBonds** - 8.4%

### Key Insights

- **Chemical Structure Dominates**: Molecular descriptors are far more predictive than categorical features
- **Polar Surface Area**: Strongest predictor, indicating the importance of intermolecular interactions
- **Molecular Size**: Weight and structural complexity significantly influence flashpoint
- **Aromatic Systems**: Ring structures play a crucial role in thermal stability

## 🛠️ Technical Details

### Dependencies

- **Data Processing**: pandas, numpy
- **Cheminformatics**: rdkit
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Development**: jupyter

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```



