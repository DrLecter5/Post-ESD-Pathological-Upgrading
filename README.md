# Pathological Upgrade Prediction Platform for Early Esophageal Cancer Post-ESD

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)

This repository contains the source code, pre-trained model weights, and anonymous patient cohorts for our multi-modal artificial intelligence online prediction tool. The platform integrates a Convolutional Neural Network (EfficientNet-B1 + CBAM) for endoscopic image analysis with clinical baseline features to accurately predict the risk of pathological upgrading after Endoscopic Submucosal Dissection (ESD).

## 🌐 Web Application (Online Platform)
You can directly access the interactive web application without any local installation:
**[Link to your deployed Streamlit Cloud app here]**

## 📂 Repository Structure
- `app.py`: The main Streamlit web application script, integrating both image feature extraction (CNNpred) and the multivariable logistic fusion model.
- `best_model.pth`: The pre-trained PyTorch weights of the Single-Stream EfficientNet-B1-CBAM model.
- `requirements.txt`: Python environment dependencies.
- `Training_Cohort_Patients.csv`: Anonymized clinical data for the training cohort (N=255).
- `Internal_Validation_Patients.csv`: Anonymized clinical data for the internal validation cohort (N=109).
- `External_Validation_Patients.csv`: Anonymized clinical data for the external validation cohort (N=44).

## 🚀 Usage Guide

### 1. Online Prediction (Single Patient)
1. Navigate to the **"单例预测" (Single Prediction)** tab on the web app.
2. **Image Upload (Optional but Recommended):** Upload a panoramic Narrow-Band Imaging (NBI) endoscopic image. The system will automatically run the PyTorch model to calculate the `CNNpred_Mean` (AI's predicted risk score based solely on visual features).
3. **Clinical Variables:** Select the clinical and endoscopic parameters, including IPCL classification (A, B1, B2, B3), ulceration, hyperemia, nodularity, lesion size, morphological type, and SIRI value.
4. Click **"开始预测" (Predict)** to obtain the real-time Fusion Model risk probability and clinical recommendations.

### 2. Batch Prediction
1. Navigate to the **"批量预测" (Batch Prediction)** tab.
2. Download the provided CSV template.
3. Fill in the clinical variables and the pre-calculated `CNNpred_Mean` values for your cohort.
4. Upload the completed file to receive an automated Excel report with the predicted probabilities for all patients.

## 🛠️ Local Deployment (For Developers)
If you wish to run this tool locally:
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
pip install -r requirements.txt
streamlit run app.py
