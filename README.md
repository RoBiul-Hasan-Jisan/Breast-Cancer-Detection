# Breast Cancer Detection App

This application uses machine learning to classify breast tumors as **Malignant** or **Benign** based on features from a breast cancer dataset.

---


## Overview
Breast cancer is one of the most common cancers worldwide. Early detection is crucial for effective treatment. This app leverages machine learning techniques to analyze tumor features and predict if the tumor is malignant (cancerous) or benign (non-cancerous).

---

## Features
- Upload breast cancer dataset in CSV format  
- Train machine learning model on the dataset  
- Evaluate model performance (accuracy, precision, recall, etc.)  
- Predict tumor type (Malignant or Benign) for new data entries  
- User-friendly interface for easy interaction  

---

## Dataset

[Uploading breast_cancer.csv…]()

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/RoBiul-Hasan-Jisan/Breast-Cancer-Detection
   cd breast-cancer-detection-app
   ```
Create and activate a virtual environment ):

```bash
python -m venv env
```
Install required dependencies:

```bash

pip install -r requirements.txt
```


Run the app:
```bash
python app.py
or
streamlit run app.py
```
Upload the dataset via the app interface.

Train the model and make predictions.
---

### Model Training
The app uses supervised learning to train a classification model. Common algorithms include:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Model performance metrics such as accuracy, precision, recall, and F1-score are displayed after training.


### Prediction
Once the model is trained, you can input new tumor data or upload a CSV file with tumor features to predict whether the tumor is malignant or benign.

### Technologies
Python 3.x

Scikit-learn

Pandas

NumPy

Streamlit 

### Output

![image](https://github.com/user-attachments/assets/7415c8ab-4fd6-49a1-95cb-25525f34d92c)

![image](https://github.com/user-attachments/assets/7ad12d93-8e9f-4926-8a63-bd3833c1018a)

![image](https://github.com/user-attachments/assets/65b62b00-bb3f-4586-9629-60c6624632e3)

![image](https://github.com/user-attachments/assets/811e8fb6-e1db-472d-a1a2-5815810bf96d)




