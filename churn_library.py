"""
Module for performing Customer Churn analysis and predictive modeling.

Author: Rohan Chakraborty
Date  : 28.08.2024
"""

# Import necessary libraries
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def save_classification_reports(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    """
    Generate and save classification reports for both training and testing data.

    Args:
        y_train (pd.Series): Actual training labels.
        y_test (pd.Series): Actual test labels.
        y_train_preds_lr (np.array): Predictions from Logistic Regression on training data.
        y_train_preds_rf (np.array): Predictions from Random Forest on training data.
        y_test_preds_lr (np.array): Predictions from Logistic Regression on test data.
        y_test_preds_rf (np.array): Predictions from Random Forest on test data.
    
    Returns:
        None
    """
    # Save Random Forest classification report
    plt.figure(figsize=(6, 6))
    plt.text(0.01, 1.25, 'Random Forest Training Results', {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, classification_report(y_train, y_train_preds_rf), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Random Forest Test Results', {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, classification_report(y_test, y_test_preds_rf), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_classification_report.png')
    plt.close()

    # Save Logistic Regression classification report
    plt.figure(figsize=(6, 6))
    plt.text(0.01, 1.25, 'Logistic Regression Training Results', {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, classification_report(y_train, y_train_preds_lr), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, 'Logistic Regression Test Results', {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, classification_report(y_test, y_test_preds_lr), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/lr_classification_report.png')
    plt.close()


def plot_feature_importance(model, x_train, output_path):
    """
    Plot and save the feature importance from the model.

    Args:
        model: Trained model object containing feature_importances_ attribute.
        x_train (pd.DataFrame): Training data used to fit the model.
        output_path (str): Path to save the feature importance plot.
    
    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [x_train.columns[i] for i in indices]

    plt.figure(figsize=(15, 7))
    plt.title("Feature Importance")
    plt.bar(range(x_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(x_train.shape[1]), feature_names, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_path}/feature_importances.png')
    plt.close()


class CustomerChurnModel:
    """
    A class for handling the entire workflow of customer churn analysis and modeling.

    Methods:
        import_data: Load data from the specified path.
        perform_eda: Conduct exploratory data analysis and save visualizations.
        encode_categorical_features: Encode categorical features using one-hot encoding.
        perform_feature_engineering: Scale features and split data into training and testing sets.
        train_and_evaluate_models: Train models, evaluate them, and save the results.
    """

    def __init__(self):
        self.data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def import_data(self, path):
        """
        Load the dataset from the provided path.

        Args:
            path (str): Path to the CSV file containing the data.
        
        Returns:
            pd.DataFrame: Loaded data as a Pandas DataFrame.
        """
        print(f"Loading data from {path}")
        self.data = pd.read_csv(path)
        self.data['Churn'] = self.data['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        self.data.drop(['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1, inplace=True)
        return self.data

    def perform_eda(self):
        """
        Perform exploratory data analysis and save the results.

        Returns:
            None
        """
        print("Performing EDA and saving figures...")
        numeric_columns = self.data.select_dtypes(include="number").columns
        categorical_columns = self.data.select_dtypes(exclude="number").columns

        for column in numeric_columns:
            plt.figure(figsize=(10, 5))
            self.data[column].hist()
            plt.title(f'{column} Distribution')
            plt.savefig(f'./images/eda/{column}_distribution.png')
            plt.close()

        for column in categorical_columns:
            plt.figure(figsize=(10, 5))
            self.data[column].value_counts(normalize=True).plot(kind='bar')
            plt.title(f'{column} Distribution')
            plt.savefig(f'./images/eda/{column}_distribution.png')
            plt.close()

        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=False, cmap='coolwarm', linewidths=2)
        plt.title('Correlation Heatmap')
        plt.savefig('./images/eda/correlation_heatmap.png')
        plt.close()

    def encode_categorical_features(self, category_columns):
        """
        Encode categorical features using one-hot encoding.

        Args:
            category_columns (list): List of column names that are categorical.
        
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        print("Encoding categorical features...")
        self.data = pd.get_dummies(self.data, columns=category_columns)
        return self.data

    def perform_feature_engineering(self, response_column, test_size=0.3):
        """
        Scale features and split the data into training and testing sets.

        Args:
            response_column (str): The target column name.
            test_size (float): The proportion of the dataset to include in the test split.
        
        Returns:
            tuple: Split data (x_train, x_test, y_train, y_test)
        """
        print("Performing feature engineering...")
        y = self.data[response_column]
        X = self.data.drop(columns=[response_column])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        return self.x_train, self.x_test, self.y_train, self.y_test

    def train_and_evaluate_models(self):
        """
        Train models, evaluate them, and save the results.

        Returns:
            None
        """
        print("Training and evaluating models...")

        # Initialize models
        rf_model = RandomForestClassifier(random_state=42)
        lr_model = LogisticRegression()

        # Hyperparameters for Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, None],
            'criterion': ['gini', 'entropy']
        }

        grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

        # Train the models
        grid_search_rf.fit(self.x_train, self.y_train)
        lr_model.fit(self.x_train, self.y_train)

        # Predictions
        y_train_preds_rf = grid_search_rf.best_estimator_.predict(self.x_train)
        y_test_preds_rf = grid_search_rf.best_estimator_.predict(self.x_test)
        y_train_preds_lr = lr_model.predict(self.x_train)
        y_test_preds_lr = lr_model.predict(self.x_test)

        # Save models
        joblib.dump(grid_search_rf.best_estimator_, './models/random_forest_model.pkl')
        joblib.dump(lr_model, './models/logistic_regression_model.pkl')

        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plot_roc_curve(lr_model, self.x_test, self.y_test, ax=plt.gca(), alpha=0.8)
        plot_roc_curve(grid_search_rf.best_estimator_, self.x_test, self.y_test, ax=plt.gca(), alpha=0.8)
        plt.savefig('./images/results/roc_curve.png')
        plt.close()

        # Generate and save classification reports
        save_classification_reports(self.y_train, self.y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

        # Plot feature importance
        plot_feature_importance(grid_search_rf.best_estimator_, pd.DataFrame(self.x_train, columns=self.data.columns[:-1]), "./images/results")


if __name__ == "__main__":
    # Define paths and parameters
    DATA_PATH = "./data/bank_data.csv"
    CATEGORICAL_COLUMNS = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    TARGET_COLUMN = 'Churn'

    # Create an instance of the model class
    churn_model = CustomerChurnModel()

    # Run the workflow
    churn_model.import_data(DATA_PATH)
    churn_model.perform_eda()
    churn_model.encode_categorical_features(CATEGORICAL_COLUMNS)
    churn_model.perform_feature_engineering(TARGET_COLUMN)
    churn_model.train_and_evaluate_models()