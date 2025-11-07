# Stress Level ML Research 🔬

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://dagshub.com/Zerol-91/Stress-Level-ML-Researching.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-blue)](https://dagshub.com/Zerol-91/Stress-Level-ML-Researching/experiments)

## 📖 Overview

This project focuses on developing machine learning models to predict anxiety levels (measured by GAD-7 scale) using survey data collected from 400+ respondents. The research compares multiple regression algorithms to identify the most accurate predictor and provides insights into factors influencing anxiety.


**Key Features:**
- Multi-dataset analysis and feature engineering
- Comprehensive model comparison (Random Forest, LightGBM, CatBoost, Linear Regression)
- MLflow experiment tracking with DagsHub integration
- SHAP analysis for model interpretability
- Real-world survey data collection and validation


## 🏗️ Project Structure
Stress-Level-ML-Researching/
├── notebooks/
│ ├── categorization_baseline.ipynb # Initial dataset exploration
│ └── preprocessing.ipynb # Main modeling notebook
├── Stress_Predict_Data_Preprocessing.ipynb # Initial dataset exploration
├── Anxiety_Level_Baseline (2).ipynb # Main modeling notebook
├── Anxiety_Level_Test_2.csv # Google Forms responses
└── README.md

## 📊 Dataset

### Survey Data Collection
Target Variable: GAD-7 anxiety score (0-21) - sum of 7 anxiety questions

Respondents: 400+ participants from various social groups
Features: 16 behavioral, lifestyle and demographic factors
Collection Method: Google Forms with clinically validated GAD-7 questionnaire
#### Target Variable (GAD-7 Questions)
All questions rated on 0-3 scale for frequency over past 2 weeks:

Feeling nervous, anxious, or on edge
Not being able to stop or control worrying
Worrying too much about different things
Trouble relaxing
Being so restless that it's hard to sit still
Becoming easily annoyed or irritable
Feeling afraid as if something awful might happen

#### Predictor Features Used:
Lifestyle & Health Factors:

Sleep duration: Average nightly sleep hours (2-14 with 0.5 increments)
Physical activity: Weekly exercise hours (0-35 with 0.5 increments)
Alcohol consumption: Frequency from "Never" to "Almost daily" (0-5 scale)
Smoking status: Non-smoker, Occasional, Daily smoker (0-2 scale)
Diet quality: Self-rated nutrition quality (1-10 with detailed descriptors)
Medication use: Current anxiety/depression medication (Yes/No)
Therapy sessions: Psychologist visits per month (0, 1, 2, 3+)
Social & Environmental Factors:
Outdoor activity: Frequency of going outside (0-4 scale)
Social satisfaction: Satisfaction with social interactions (1-5 scale)
Work/study environment: Satisfaction with current environment (1-5 scale)
Social support: Perceived support from close ones (1-5 scale)
Significant life events: Major stressful events in past year (Yes/No)
Behavioral & Demographic Factors:
Chronotype: Morning person, Neutral, Night owl
Phone usage: Daily hours on phone excluding work (0-24 with 0.5 increments)
Gender: Male, Female, Prefer not to say
Age: 16-75 years


## 🧪 Experiments & Models

### Model Comparison
| Model | MAE | R² | RMSE | Status |
|-------|-----|----|------|--------|
| CatBoost (Bagging + Log Transform) | **[2.47]** | **[0.53]** | **[3.03]** | 🏆 Best |

### MLflow Tracking
All experiments are logged with MLflow and available on [DagsHub](https://dagshub.com/Zerol-91/Stress-Level-ML-Researching/experiments):
