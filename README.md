# 🧠 Customer Churn Prediction — Machine Learning Pipeline  
**Day 19 of the 64-Day Machine Learning Challenge**

This project implements an **end-to-end churn prediction system** using classical machine learning techniques.  
It demonstrates how real businesses can identify customers at risk of leaving by analyzing behavioral, demographic, and service-usage patterns.

The solution includes **data preprocessing**, **feature engineering**, **model training**, **evaluation**, and **artifact saving**, making this a production-ready baseline workflow.

---

## ⚙️ 1. Problem Overview

Customer churn refers to the percentage of customers who stop using a product or service.  
Predicting churn helps companies:

- Reduce revenue loss  
- Identify high-risk customers  
- Improve retention strategies  
- Optimize targeted marketing  

This project uses the **Telco Customer Churn dataset** (or an auto-generated demo dataset if missing) to build a churn prediction model.

---

## 🏗️ 2. System Architecture

```
                ┌────────────────────────┐
                │   Raw Input Dataset     │
                └─────────────┬──────────┘
                              ▼
                   Data Preprocessing
    ┌────────────────────────────────────────────────────┐
    │ - Missing value handling                           │
    │ - Categorical encoding (One-Hot / Ordinal)         │
    │ - Numerical scaling                                │
    │ - Train-test split                                 │
    └────────────────────────────────────────────────────┘
                              ▼
                     Model Training
    ┌────────────────────────────────────────────────────┐
    │ - Logistic Regression                              │
    │ - Random Forest Classifier                         │
    │ - Hyperparameter-ready pipeline                    │
    └────────────────────────────────────────────────────┘
                              ▼
                      Model Evaluation
    ┌────────────────────────────────────────────────────┐
    │ - Accuracy                                         │
    │ - Precision / Recall / F1-score                    │
    │ - Confusion Matrix                                 │
    │ - Feature Importance                               │
    └────────────────────────────────────────────────────┘
                              ▼
                     Artifact Generation
    ┌────────────────────────────────────────────────────┐
    │ - churn_model.pkl                                   │
    │ - preprocessor.pkl                                   │
    │ - confusion_matrix.png                               │
    │ - feature_importance.png                             │
    └────────────────────────────────────────────────────┘
```

---

## 📂 3. Repository Structure

```
├── run_churn.py               # Main execution script
├── data_utils.py              # Load & preprocess raw data
├── model_utils.py             # Model training logic + wrappers
├── viz_utils.py               # Visualization utilities
│
├── telco_churn.csv            # Dataset (auto-generated if missing)
├── churn_model.pkl            # Saved model
├── preprocessor.pkl           # Saved preprocessing pipeline
│
├── confusion_matrix.png       # Evaluation visualization
├── feature_importance.png     # Feature significance plot
│
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

---

## 🧹 4. Data Preprocessing Details

The preprocessing pipeline includes:

### **Categorical Variables**
- One-Hot Encoding for multi-class categorical features  
- Ordinal encoding where meaningful order exists  
- Handling of “Yes/No” binary fields

### **Numerical Variables**
- StandardScaler applied to continuous columns  
- Automatic detection of numerical columns  
- Outlier-tolerant transformations

### **Train–Test Split**
- 80/20 split  
- Stratified splitting to preserve churn distribution  

---

## 🤖 5. Machine Learning Models

The project uses a modular structure allowing quick switching between models:

### **Implemented Models**
- **Logistic Regression**  
- **Random Forest Classifier**

### **Easily Extendable**
You can plug in:

- XGBoost  
- LightGBM  
- CatBoost  
- SVM  
- Neural Networks  

The `train_model()` function handles any scikit-learn compatible model.

---

## 📊 6. Evaluation Metrics

After model training, the system generates:

- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Classification Report**
- **Confusion Matrix Heatmap**
- **Feature Importance Bar Chart**

These results help determine what influences customer churn and how well the model generalizes.

---

## 📝 7. How to Run the Project

### **1️⃣ Install Dependencies**
```
pip install -r requirements.txt
```

### **2️⃣ Execute Churn Pipeline**
```
python3 run_churn.py
```

### **3️⃣ Outputs Generated**
After running, you will see:

```
churn_model.pkl
preprocessor.pkl
confusion_matrix.png
feature_importance.png
```

---

## 📈 8. Key Insights from Model

- Categorical service-related fields significantly influence churn  
- Contract type and monthly charges are strong predictors  
- Tenure often negatively correlates with churn  
- Random Forest outperforms Logistic Regression for baseline prediction  

---

## 🧭 9. Future Improvements

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- SMOTE balancing for imbalanced churn labels  
- Feature selection using SHAP values  
- Deployment as a FastAPI/Flask web service  
- Monitor live churn probabilities

---

## 📜 License
MIT License — free for personal and commercial use.

---

## 🤝 Contributions
Pull requests and feature suggestions are welcome.