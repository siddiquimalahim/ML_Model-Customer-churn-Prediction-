# Customer Churn Prediction

## Project Overview

This project uses Machine Learning to predict whether a customer is likely
to churn.

The project was developed as an end-to-end Machine Learning classification
project using Python and Scikit-Learn.

## Dataset

The dataset contains customer demographic, billing, service, support and
usage information.

The target variable is:

- `churn`

where:

- `Yes` = customer churned
- `No` = customer did not churn

## Machine Learning Models

Two classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## Final Model

Logistic Regression was selected as the final model based on its overall
test-set performance.

The complete preprocessing and trained model were saved as:

`churn_pipeline.pkl`

## Streamlit Application

The Streamlit application allows users to enter customer information and
receive a churn prediction.

The application also displays the estimated churn probability.

## Project Files

- `customer_churn_data.csv` - Dataset
- `model_training.ipynb` - Jupyter Notebook
- `churn_pipeline.pkl` - Trained Machine Learning pipeline
- `app.py` - Streamlit application
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation

## Running the Application

Install the required packages:

```bash
pip install -r requirements.txt