# 🫀 Heart Disease Prediction Project

## Overview
This project implements a machine learning system to predict heart disease risk using the Cleveland Heart Disease dataset. The system achieves **88.5% accuracy** and **95.2% ROC-AUC** using Random Forest classification.

## 🎯 Project Goals
- Predict heart disease risk from patient medical data
- Implement multiple ML algorithms and compare performance
- Create an interactive web application using Streamlit
- Deploy the model for real-world use

## �� Dataset
- **Source**: Cleveland Heart Disease Database (UCI ML Repository)
- **Features**: 13 medical attributes (age, sex, chest pain type, blood pressure, etc.)
- **Target**: Binary classification (0 = no disease, 1 = disease present)
- **Size**: 303 patients

## 🚀 Features
- **Data Preprocessing**: Handling missing values, feature scaling
- **Feature Selection**: PCA, Random Forest importance, RFE, Chi-square test
- **ML Models**: Logistic Regression, Decision Tree, Random Forest, SVM
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Clustering Analysis**: K-Means clustering for patient segmentation
- **Web Application**: Interactive Streamlit UI for predictions

## �� Project Structure


## ��️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset to `data/` folder

## �� Usage

### Run the Streamlit App
```bash
streamlit run app/heart_disease_app.py
```

The app will open at `http://localhost:8501`

### Use the Jupyter Notebook
Open `notebooks/heart_project.ipynb` to explore the complete analysis

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Random Forest | 88.5% | 95.2% | 87.3% |
| Logistic Regression | 86.9% | 95.1% | 86.1% |
| SVM | 85.3% | 94.4% | 84.7% |
| Decision Tree | 73.8% | 74.4% | 73.2% |

## 🔍 Key Findings

### Top 5 Important Features:
1. **Maximum Heart Rate (thalach)** - 13.8%
2. **ST Depression (oldpeak)** - 12.2%
3. **Cholesterol (chol)** - 11.6%
4. **Age** - 11.4%
5. **Number of Major Vessels (ca)** - 9.7%

### Clustering Results:
- **3 patient clusters** identified using K-Means
- Each cluster shows different heart disease patterns
- Useful for patient segmentation and personalized care

## 🌐 Deployment

### Local Deployment
```bash
streamlit run app/heart_disease_app.py
```

### Cloud Deployment
The app can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/GCP
- Docker containers

## 📚 Technical Details

### Data Preprocessing
- Standardization using StandardScaler
- PCA for dimensionality reduction
- Missing value imputation

### Feature Selection Methods
- Recursive Feature Elimination (RFE)
- Chi-square test
- Random Forest importance

### Model Training
- 80/20 train-test split
- Stratified sampling
- Cross-validation for hyperparameter tuning

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details

## 🙏 Acknowledgments
- Cleveland Clinic Foundation for the dataset
- UCI Machine Learning Repository
- Scikit-learn and Streamlit communities

## �� Contact
- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername]

---
⭐ **Star this repository if you find it helpful!**
