# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Project Description
G'day! Welcome to the Predicting Customer Churn project, where we're diving deep into the world of banking to figure out which customers might be heading for the door. In the fast-paced world of finance, keeping customers happy and sticking around is as important as a good barbie on a sunny day. Losing customers, or churn as we call it, can seriously dent a bank's bottom line. That's why we're rolling up our sleeves and using some top-notch data science to spot the signs of customer churn before it happens.

This project is all about getting ahead of the game. We’re working with a dataset that includes a bunch of info about customers, and our mission is to use that data to predict who might be waving goodbye. The idea is simple: by knowing who’s at risk of leaving, the bank can take action to keep them around – like offering a sweet deal or improving the service they’re getting.

Here’s how we’re cracking it:

- Data Wrangling: First, we’ve got to get our hands on the data and give it a good clean.
- Exploratory Data Analysis (EDA): Then, we’ll have a stickybeak at the data to see what it’s telling us – checking out distributions, spotting trends, and uncovering correlations.
- Feature Engineering: We’ll whip the data into shape with some one-hot encoding for categorical features, standardize the numerical data, and split it into training and testing sets.
- Modeling: Time to get serious. We’re training up two models – Logistic Regression and Random Forest – and putting them through their paces. We’ll tune them up, evaluate them with ROC curves, and see how they stack up.
- By the end of this project, we’ll have a solid model that can help banks hang on to their customers. It’s all about staying ahead of the competition and making sure customers are happy as Larry. So, grab your sunnies and let’s get to work!

## Files and data description
```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
├── Guide.ipynb
├── images
│   ├── eda
│   └── results
├── logs
├── models
├── __pycache__
├── README.md
└── requirements.txt
```

## Running Files

### Dependencies
```
autopep8==1.5.7
joblib==0.11
matplotlib==2.1.0
numpy==1.12.1
pandas==0.23.3
pylint==2.9.6
scikit-learn==0.22
seaborn==0.8.1
```
```
pip install -r requirements.txt
```

### Linting

Score churn_library = 8.05/10
Score churn_script_logging_and_tests.py = 7.90/10 << Needs some work
```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```
### Testing and Logging

```
ipython churn_library.py
```
### Modeling

```
ipython churn_library.py
```
