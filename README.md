# 🧬 Breast Cancer Classification – Model Building

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Data Analysis](https://img.shields.io/badge/DataAnalysis-EDA-orange)
![Visualization](https://img.shields.io/badge/Visualization-Seaborn-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

A supervised machine learning project focused on implementing and comparing
multiple **classification algorithms** using a real-world medical dataset.

---

## 🚀 Run Notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ACKyXvSBzgVsd2WDiHB7KiHpSyWaMfVo)

---

## 📘 Project Overview

- This project applies various **classification algorithms** to the **Breast Cancer dataset**
  available in the `sklearn` library.
- The dataset consists of **569 samples with 30 numerical features** describing tumor
  characteristics, and a binary target variable indicating whether the tumor is **Benign** or **Malignant**.
- The primary goal is to build, evaluate, and compare multiple classification models
  to identify the most effective algorithm for this dataset.

---

## 🎯 Objective

The project includes:

🔹 Dataset loading and preprocessing  
🔹 Feature scaling and data preparation  
🔹 Implementation of multiple classification algorithms  
🔹 Model evaluation and comparison  
🔹 Interpretation of results  
🔹 Clean and reproducible Google Colab Notebook  

---

## 📂 Dataset Description

The dataset is sourced from `sklearn.datasets.load_breast_cancer`.

| Component | Description |
|---------|-------------|
| Samples | 569 |
| Features | 30 numerical features |
| Target Classes | Benign (0), Malignant (1) |

---

## 🧹 Preprocessing Steps

✔ Dataset loading using `load_breast_cancer()`  
✔ Converted into Pandas DataFrame  
✔ Checked missing values (none found)  
✔ Checked duplicates (none found)  
✔ Train–Test split  
✔ Feature scaling using `StandardScaler`  

---

## 🤖 Classification Algorithms Implemented

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- k-Nearest Neighbors (k-NN)  

---

## 📊 Model Evaluation Metrics

- Accuracy Score  
- Confusion Matrix  
- Precision  
- Recall  
- F1-score  

---

## 📊 Model Performance Comparison

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 0.982456 |
| SVM (RBF)              | 0.982456 |
| k-NN                   | 0.956140 |
| Random Forest          | 0.956140 |
| Pruned Decision Tree   | 0.921053 |
| Decision Tree          | 0.912281 |

---

## 🏆 Best and Worst Performing Models

**Best Model:** Logistic Regression & SVM (RBF) → **0.982**  
**Worst Model:** Decision Tree → **0.912**

---

## 🧠 Key Observations

✔ Feature scaling improved SVM and k-NN performance  
✔ Logistic Regression & SVM performed best  
✔ Random Forest provided good generalization  
✔ Decision Tree showed overfitting  
✔ Pruning improved Decision Tree  

---

## 🛠 Tech Stack

| Tool | Purpose |
|----|--------|
| Python | Programming |
| Pandas | Data handling |
| NumPy | Computation |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| Scikit-learn | ML models |
| Google Colab | Development |

---

## 📁 Repository Structure

Classification-Algorithms-Model-Building/

│  

├── Classification_Algorithms_Model_Building.ipynb  

├── README.md  

---

## 🚀 How to Run the Project

1️⃣ Open the notebook using the Colab link above  
2️⃣ Run all cells sequentially  
3️⃣ View model results and evaluation  

---

## 📌 Academic Submission

This project was created as part of a **Machine Learning academic assignment**, demonstrating the implementation and comparison of multiple classification algorithms using a real-world medical dataset.

---

## ⚠️ Limitations

- Small dataset size  
- Limited features  
- Risk of overfitting  
- No extensive hyperparameter tuning  
- Not tested on external datasets  

---

## 📌 Future Enhancements

- Apply GridSearchCV tuning  
- Use advanced models (XGBoost, LightGBM)  
- Add cross-validation  
- Build Streamlit web app  
- Improve visualization dashboard  

---

## 👤 Author

**Name:** Laya Mary Joy  

**Organization:** Entri Elevate  

**Date:** January 19, 2026  

---

## ⭐ Acknowledgment

Thanks to **Entri Elevate** for guidance and support throughout this project.

---
Thanks to **Entri Elevate** for guidance and support.

---
