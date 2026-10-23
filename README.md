# Automated IT & Financial Support Ticket Routing & SLA Predictor

An end-to-end Machine Learning and MLOps pipeline designed to automatically classify customer support tickets into functional product domains and predict Service Level Agreement (SLA) breach risks. Built using public complaint logs from the Consumer Financial Protection Bureau (CFPB).

---

## 📋 Project Overview & Pipeline Architecture

This project implements a complete, end-to-end NLP and machine learning pipeline that ingests raw consumer complaint records, preprocesses text metadata, constructs TF-IDF features, and trains multi-class classification and binary risk prediction models.

---

## 🛠️ Pipeline Modules & Workflow

### **Module 1: Data Ingestion, EDA & Semantic Feature Engineering**
* **Ingestion & Downsampling:** Ingests raw CFPB complaint logs (52,702 records across 15 features) and filters them into 4 primary product domains:
  * *Vehicle loan or lease*
  * *Payday loan, title loan, personal loan, or advance loan*
  * *Prepaid card*
  * *Debt or credit management*
  Executes stratified downsampling to create a balanced dataset of **25,000 records**.
* **Exploratory Data Analysis (EDA):** Generates and exports distribution plots (`eda_product_distribution.png`) and SLA timeliness breakdown charts across submission channels (`eda_sla_by_channel.png`).
* **Data Cleansing & Imputation:** Handles missing values across `Sub-issue`, `Company`, `Submitted via`, and `Timely response?` by imputing `'Unknown'` placeholders and stripping whitespace.
* **Semantic Feature Fusion:** Constructs a unified feature (`combined_text`) formatted as:
  `"Sub-issue: {Sub-issue} | Company: {Company} | Submitted via: {Submitted via}"`
* **Target Encoding & Data Splitting:** Converts `Timely response?` into a binary integer target (`1` for On-Time, `0` for SLA Breach). Performs a 2-stage stratified split:
  * **Training Set:** 20,000 rows (80%)
  * **Validation Set:** 2,500 rows (10%)
  * **Test Set:** 2,500 rows (10%)
* **Text Vectorization:** Fits a `TfidfVectorizer` (top 5,000 features limit, English stop-word removal) on the training set yielding a sparse feature matrix shape of `(20000, 1353)`.

### **Module 2: Multi-Class Support Ticket Routing Engine**
* **Random Forest Classifier:** Ensembled 100-tree baseline model (`n_estimators=100`, `random_state=42`) achieving **94.64% validation accuracy**.
* **XGBoost Classifier:** Multi-class gradient boosting model (`max_depth=6`, `learning_rate=0.1`, `mlogloss` loss function) trained on label-encoded target classes achieving **93.76% validation accuracy**.
* **TensorFlow Neural Network (MLP):** Sequential Deep Learning network featuring:
  * Dense Input Layer (128 units, ReLU activation) + 30% Dropout Regularization
  * Hidden Dense Layer (64 units, ReLU activation) + 30% Dropout Regularization
  * Softmax Output Layer (4 domain classes)
  * Trained over **10 epochs** with batch size of **64** using Adam optimizer and sparse categorical crossentropy loss, achieving **95.12% validation accuracy**.

### **Module 3: Predictive SLA Resolution Engine**
* **XGBoost SLA Risk Classifier:** Binary classification model (`max_depth=5`, `learning_rate=0.1`, `n_estimators=100`) predicting whether a complaint will meet its resolution timeline or experience an SLA breach, achieving **96.72% validation accuracy**.

---

## 📈 Model Performance & Metrics

Evaluated on the 2,500-record validation set partition:

| Pipeline Module | Model Architecture | Target Variable | Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Module 2: Ticket Routing** | Random Forest | Product Domain (Multi-Class) | Validation Accuracy | **94.64%** |
| **Module 2: Ticket Routing** | XGBoost Classifier | Product Domain (Multi-Class) | Validation Accuracy | **93.76%** |
| **Module 2: Ticket Routing** | TensorFlow Deep MLP | Product Domain (Multi-Class) | Validation Accuracy | **95.12%** |
| **Module 3: SLA Risk Engine** | XGBoost Risk Classifier | SLA Breach (`sla_target`: 1/0) | Validation Accuracy | **96.72%** |

---
