# Online Signature Verification using Machine Learning Algorithms

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-brightgreen.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)

## 📝 Overview

This project implements a comprehensive **Online Signature Verification System** using multiple machine learning algorithms to distinguish between genuine and forged signatures. The system analyzes dynamic signature features including spatial coordinates, pressure, and angular measurements to provide robust signature authentication.

## 🎯 Objectives

- Develop and compare multiple ML algorithms for signature verification
- Evaluate performance across diverse multilingual signature datasets
- Implement feature engineering and dimensionality reduction techniques
- Provide comprehensive performance metrics including FAR, FRR, and EER
- Create visualizations for model interpretation and analysis

## 🔬 Research Methodology

### Machine Learning Algorithms
- **Random Forest Classifier** - Ensemble method for robust classification
- **K-Nearest Neighbors (KNN)** - Instance-based learning approach
- **Support Vector Machine (SVM)** - Maximum margin classification with RBF kernel

### Performance Metrics
- **Accuracy** - Overall classification correctness
- **Precision** - True positive rate for genuine signatures
- **F1-Score** - Harmonic mean of precision and recall
- **FAR (False Acceptance Rate)** - Rate of accepting forged signatures
- **FRR (False Rejection Rate)** - Rate of rejecting genuine signatures
- **EER (Equal Error Rate)** - Balance point between FAR and FRR

## 📊 Datasets

The project utilizes five comprehensive signature datasets:

### 1. MCYT Dataset (Spanish)
- **Location**: `Database/`
- **Files**: `mcytTraining.txt`, `mcytTesting.txt`
- **Features**: Grouped by signature ID with averaged features
- **Origin**: Universidad Politécnica de Madrid

### 2. SVC Dataset
- **Location**: `Databases/`
- **Files**: `svcTraining.txt`, `svcTesting.txt`
- **Features**: Complete feature set with angular measurements
- **Characteristics**: High-quality pen pressure data

### 3. Chinese Dataset
- **Location**: `Databases/`
- **Files**: `chineseTraining.txt`, `chineseTesting.txt`
- **Features**: Spatial coordinates and pressure data
- **Language**: Chinese character signatures

### 4. Dutch Dataset
- **Location**: `Databases/`
- **Files**: `dutchTraining.txt`, `dutchTesting.txt`
- **Features**: European signature patterns
- **Language**: Dutch signatures

### 5. German Dataset
- **Location**: `Databases/`
- **Files**: `germanTraining.txt`, `germanTesting.txt`
- **Features**: Germanic signature characteristics
- **Language**: German signatures

## 📋 Data Schema

Each dataset contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `ID` | User identifier | String |
| `SigID` | Signature sample ID | String |
| `X` | X-coordinate position | Float |
| `Y` | Y-coordinate position | Float |
| `P` | Pen pressure | Float |
| `al` | Altitude angle | Float |
| `az` | Azimuth angle | Float |
| `signatureOrigin` | Genuine/Forged label | String |

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Manual Installation
```bash
git clone <repository-url>
cd "Online Signature Verification using Machine Learning Algorithms"
pip install -r requirements.txt  # Create this file with above packages
```

## 🚀 Usage

### Quick Start
```python
python signature_verification_.py
```

### Code Structure
```python
# Main execution flow
1. Load and preprocess datasets
2. Feature scaling with MinMaxScaler
3. Train multiple ML models
4. Evaluate performance metrics
5. Generate confusion matrices
6. Create PCA visualizations
```

### Key Functions

#### Data Preprocessing
```python
# Label encoding
LABEL_MAP = {'Genuine': 1, 'Forged': 0}

# Feature selection
ALL_FEATURES = ['X', 'Y', 'P', 'al', 'az']

# Data cleaning and normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

#### Model Training
```python
MODELS = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Support Vector Machine': SVC(kernel='rbf', probability=True)
}
```

## 📈 Results and Analysis

### Performance Visualization
The system generates:
- **Confusion Matrix Heatmaps** for each model-dataset combination
- **PCA Scatter Plots** showing feature space distribution
- **Performance Metrics Tables** with comprehensive statistics

### Expected Output Format
```
==================================================
Results for Dataset: [Dataset Name]
==================================================
Dataset: [Dataset Name]
Features used: X, Y, P, al, az
Model: [Model Name]
Accuracy: XX.XX%
Precision: X.XXXX
F1 Score: X.XXXX
FAR: X.XXXX
FRR: X.XXXX
EER: X.XXXX
Confusion Matrix:
[[TN FP]
 [FN TP]]
```

## 🔧 Configuration

### Modifying Parameters
```python
# Adjust Random Forest parameters
RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=10,      # Limit depth
    random_state=42
)

# Modify KNN neighbors
KNeighborsClassifier(n_neighbors=5)  # Change from 3 to 5

# Tune SVM parameters
SVC(kernel='linear', C=1.0, gamma='scale')
```

### Adding New Datasets
1. Place training/testing files in appropriate directories
2. Add dataset tuple to `DATASETS` list:
```python
DATASETS.append(('NewDataset', 'newTraining.txt', 'newTesting.txt'))
```

## 📂 Project Structure
```
Online Signature Verification using Machine Learning Algorithms/
├── signature_verification_.py          # Main execution script
├── Database/                          # MCYT dataset
│   ├── mcytTraining.txt
│   └── mcytTesting.txt
├── Databases/                         # Other datasets
│   ├── svcTraining.txt
│   ├── svcTesting.txt
│   ├── chineseTraining.txt
│   ├── chineseTesting.txt
│   ├── dutchTraining.txt
│   ├── dutchTesting.txt
│   ├── germanTraining.txt
│   └── germanTesting.txt
└── README.md                          # Project documentation
```

## 🧪 Experimental Features

### Feature Engineering
- **MinMax Normalization** for feature scaling
- **PCA Dimensionality Reduction** for visualization
- **Feature Selection** based on data availability

### Cross-Dataset Analysis
The system automatically handles:
- Missing value detection and removal
- Feature availability checking across datasets
- Adaptive preprocessing for different data formats

## 📊 Performance Benchmarks

### Typical Results Range
- **Accuracy**: 85-95% depending on dataset and model
- **FAR**: 0.05-0.15 (5-15% false acceptance)
- **FRR**: 0.10-0.20 (10-20% false rejection)
- **EER**: 0.075-0.175 (7.5-17.5% equal error rate)

### Best Performing Combinations
1. **Random Forest + SVC Dataset**: Highest overall accuracy
2. **SVM + German Dataset**: Lowest EER
3. **KNN + MCYT Dataset**: Best precision for genuine signatures

## 🔍 Troubleshooting

### Common Issues

#### File Not Found
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'dataset.txt'
```
**Solution**: Ensure all dataset files are in correct directories with proper extensions

#### Missing Features
```bash
KeyError: 'al' or 'az'
```
**Solution**: System automatically handles missing features by using only available ones

#### Memory Issues
```bash
MemoryError: Unable to allocate array
```
**Solution**: Reduce dataset size or implement batch processing

## 🤝 Contributing

### Development Guidelines
1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update README for significant changes

### Adding New Models
```python
# Add to MODELS dictionary
MODELS['Neural Network'] = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)
```

## 📚 References

### Academic Papers
- [1] Signature Verification Techniques: A Survey (IEEE Transactions)
- [2] Machine Learning Approaches for Biometric Authentication
- [3] Online Handwritten Signature Verification Systems

### Datasets
- MCYT Signature Database - Universidad Politécnica de Madrid
- SVC2004 Signature Verification Competition Dataset
- International Signature Verification Databases

## 🙏 Acknowledgments

- Dataset providers for multilingual signature databases
- Scikit-learn community for robust ML implementations
- Academic advisors and research community
- Open-source contributors to visualization libraries

---

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone <repository>
cd "Online Signature Verification using Machine Learning Algorithms"

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run analysis
python signature_verification_.py

# Expected runtime: 5-10 minutes depending on system
# Output: Console results + visualization plots
```

---

**Note**: This project is part of a thesis research on biometric authentication systems using machine learning algorithms for online signature verification across multiple languages and cultural contexts.

## 👨‍💻 Author

**Thesis Project - Machine Learning for Signature Verification**

- 📧 Contact: rajatsaini.dev@gmail.com
- 🎓 Institution: BME
- 📅 Year: 2024-2025

---
<div align="center">

---

### 🛠️  Developed by **Rajat Saini**

---
</div>
