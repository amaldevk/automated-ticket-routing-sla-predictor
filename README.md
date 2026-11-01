# 🛡️ Enterprise IT & Financial Support Triage & SLA Risk Command Center

An end-to-end MLOps and production-grade application designed to automate customer support ticket routing and predict Service Level Agreement (SLA) timeliness risk in real-time using Deep Learning, Gradient Boosting, and an interactive web command center.

---

## 🌟 Project Overview

In large enterprise and banking environments, manually reviewing and routing incoming customer complaints creates severe operational bottlenecks and risks breaching contractual Service Level Agreements (SLAs). This project provides an automated, AI-driven triage command center featuring:
1. **Automated Ticket Routing:** Employs a TensorFlow Multi-Layer Perceptron (MLP) deep neural network alongside TF-IDF feature embeddings to classify support tickets into key product domains.
2. **Predictive SLA Risk Scoring:** Utilizes an optimized XGBoost classifier with custom class-imbalance weighting to calculate prompt resolution probabilities and flag high-risk tickets before deadlines are violated.
3. **Interactive Executive Command Center:** Integrates a high-performance FastAPI backend with a polished Streamlit frontend dashboard, supporting real-time telemetry, risk status badge mapping, and raw feature payload inspection.

---

## ⚙️ Technical Architecture & Tech Stack

* **Machine Learning & Deep Learning:** TensorFlow / Keras (MLP Neural Network), XGBoost, Random Forest, Scikit-Learn.
* **Natural Language Processing (NLP):** `TfidfVectorizer` (max features: 5,000, English stop words removal) with rigorous string schema formatting (`Sub-issue: {text} | Company: {name} | Submitted via: {channel}`).
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations) for global feature importance and instance-level ticket audits.
* **Backend API:** FastAPI with automated Pydantic schema validation.
* **Frontend UI:** Streamlit layout with custom responsive container cards and dynamic 3-tier SLA alert banners.
* **Model Optimization:** Precision-Recall curve optimization (empirically derived optimal decision threshold set at **`0.8256`**).

---

## 📊 Dataset & Model Performance Summary

The system is trained on a stratified 25,000-record downsampled subset of the Consumer Financial Protection Bureau (CFPB) complaints dataset spanning four core product categories:
* *Vehicle loan or lease*
* *Payday loan, title loan, personal loan, or advance loan*
* *Prepaid card*
* *Debt or credit management*

### Model Validation Results
* **Random Forest Classifier:** `0.9476` Validation Accuracy
* **XGBoost Classifier:** `0.9376` Validation Accuracy
* **TensorFlow MLP Deep Neural Network:** **`0.9516` Validation Accuracy** (Selected production routing model)
* **SLA Risk Model ROC-AUC:** `0.8922`
* **Empirically Derived Optimal Decision Threshold:** `0.8256`

---


### 🚦 3-Tier Enterprise SLA Status Mapping

To bridge raw model probabilities with actionable operational workflows, the application translates scores into three distinct priority tiers:

1. **🟢 Normal / On Track :** High confidence; ticket is progressing safely within standard turnaround times.
2. **🟡 Warning / At Risk :** Moderate confidence; ticket is within tolerance but requires monitoring.
3. **🔴 Breached / Critical :** Below optimal threshold; high likelihood of exceeding SLA deadlines unless immediately escalated or reassigned.

---

