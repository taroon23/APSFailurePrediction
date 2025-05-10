# ⚙️ APS Failure Detection in Scania Trucks

This project implements a machine learning pipeline to detect rare but critical Air Pressure System (APS) failures in Scania trucks. Using tree-based classifiers and advanced imbalance correction methods, the project highlights effective approaches to real-time predictive maintenance using high-dimensional sensor data.

---

## 📌 Project Overview

The classification task is based on the [APS Failure at Scania Trucks dataset](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks). The data is extremely imbalanced, with only ~1.6% of records belonging to the positive (failure) class. The pipeline includes:

- **Data imputation** to handle missing values
- **Feature selection** using Coefficient of Variation (CV)
- **Model training** with:
  - Random Forest
  - XGBoost with L1-penalized logistic model trees
- **Imbalance mitigation** using:
  - Class weights
  - SMOTE (Synthetic Minority Over-sampling Technique)
- **Evaluation metrics**: Confusion matrix, ROC curve, AUC, misclassification rate

---

## 📁 Repository Structure

```
.
├── notebook/
│   └── APSFailurePrediction.ipynb      # Full analysis and modeling notebook
│
├── data/
│   ├── aps_failure_training_set.csv    # Training data
│   ├── aps_failure_test_set.csv        # Test data
│   └── aps_failure_description.txt     # Attribute and label descriptions
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aps-failure-detection.git
cd aps-failure-detection
```

### 2. Install Requirements
Ensure Python 3.7+ is installed. Then install required packages:
```bash
pip install -r requirements.txt
```

---

## ✅ Results & Highlights

- Random Forest and XGBoost classifiers were trained and tested.
- SMOTE + XGBoost yielded the best recall and AUC on minority (failure) class.
- Model performance validated via cross-validation and held-out test data.
- Demonstrates a scalable approach to rare-event detection in industrial systems.

---

## 🏁 Conclusion

This project showcases how ensemble methods and synthetic oversampling techniques like SMOTE can effectively handle severe class imbalance in industrial fault detection scenarios. The trained models can serve as a baseline for real-time predictive maintenance frameworks in manufacturing and automotive diagnostics.

---

## 📚 References

- [APS Failure Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks)
- Scikit-learn, Imbalanced-learn, XGBoost, Pandas, NumPy

---

## ✍️ Author

Taroon Ganesh  
Graduate Student, Data Science  
University of Southern California  
