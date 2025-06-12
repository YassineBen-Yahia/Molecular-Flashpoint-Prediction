# 🔬 Molecular Flashpoint Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-orange.svg)](https://scikit-learn.org)
[![RDKit](https://img.shields.io/badge/RDKit-2025.3.2-green.svg)](https://rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project that predicts molecular flashpoint values using chemical structure data (SMILES strings) and various molecular descriptors. This project demonstrates advanced feature engineering techniques for cheminformatics and implements robust machine learning pipelines for molecular property prediction.

## 🎯 Project Overview

Flashpoint is a critical safety parameter in chemistry and chemical engineering, representing the lowest temperature at which a volatile liquid can vaporize to form an ignitable mixture with air. This project leverages machine learning to predict flashpoint values from molecular structure data, enabling rapid screening of chemical compounds for safety assessment.

### 🌟 Key Features

- **Advanced Chemical Feature Engineering**: Convert SMILES strings to molecular descriptors using RDKit
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

### Sample Data Structure
```
compound                    | flashpoint | smiles      | is_acid | source
1-aminopropan-2-ol         | 350.15     | CC(CN)O     | 0       | pubchem
1-chloro-2,4-dinitrobenzene| 467.15     | C1=CC(=C... | 0       | pubchem
```

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

- **Built-in Random Forest Importance**: Based on mean decrease in impurity
- **Permutation Importance**: Measures actual impact on model performance
- **Threshold-based Selection**: Remove features with minimal predictive power

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
├── 📂 src/
│   ├── data_preprocessing.ipynb  # Data cleaning & feature engineering
│   └── train_model.ipynb         # Model training & evaluation
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
   git clone <repository-url>
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

2. **Model Training** (`src/train_model.ipynb`)
   - Train Random Forest model
   - Perform feature importance analysis
   - Evaluate model performance
   - Compare full vs. reduced models

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

## 📝 Future Improvements

- [ ] **Advanced Models**: Implement XGBoost, Neural Networks
- [ ] **Hyperparameter Tuning**: Grid search optimization
- [ ] **Feature Engineering**: Additional molecular descriptors
- [ ] **Cross-Validation**: K-fold validation strategy
- [ ] **Ensemble Methods**: Combine multiple algorithms
- [ ] **Web Interface**: Deploy model as web application

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RDKit Community** for excellent cheminformatics tools
- **Scikit-learn** for robust machine learning algorithms
- **Chemical Database Contributors** for providing high-quality datasets

## 📬 Contact

For questions or collaboration opportunities, please reach out!

---

⭐ **Star this repository if you found it helpful!** ⭐
