"""
Module for testing functions from churn_library.py

Author: Rohan Chakraborty
Date  : 28.08.2024
"""

import os
import logging
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def test_import(import_function):
    """
    Test the import_data function from churn_library.py

    Args:
        import_function (function): The import_data function to be tested.
    
    Returns:
        None
    """
    try:
        df = import_function("./data/bank_data.csv")
        logging.info("test_import: SUCCESS - Data loaded successfully.")
    except FileNotFoundError as err:
        logging.error("test_import: FAILURE - The specified file was not found.")
        raise err

    try:
        assert df.shape[0] > 0 and df.shape[1] > 0
    except AssertionError as err:
        logging.error("test_import: FAILURE - Dataframe is empty or not structured correctly.")
        raise err


def test_eda(eda_function):
    """
    Test the perform_eda function from churn_library.py

    Args:
        eda_function (function): The perform_eda function to be tested.
    
    Returns:
        None
    """
    try:
        eda_function()
        logging.info("test_eda: SUCCESS - EDA performed successfully.")
    except Exception as err:
        logging.error(f"test_eda: FAILURE - Error during EDA: {err}")
        raise err


def test_encoder_helper(encoder_function, categorical_columns):
    """
    Test the encode_categorical_features function from churn_library.py

    Args:
        encoder_function (function): The encode_categorical_features function to be tested.
        categorical_columns (list): List of categorical column names to encode.
    
    Returns:
        None
    """
    try:
        encoder_function(categorical_columns)
        logging.info("test_encoder_helper: SUCCESS - Encoding applied successfully.")
    except KeyError as err:
        logging.error("test_encoder_helper: FAILURE - A column specified does not exist in the dataframe.")
        raise err

    try:
        assert isinstance(categorical_columns, list) and len(categorical_columns) > 0
    except AssertionError as err:
        logging.error("test_encoder_helper: FAILURE - The input should be a non-empty list of column names.")
        raise err


def test_perform_feature_engineering(engineering_function, response_column):
    """
    Test the perform_feature_engineering function from churn_library.py

    Args:
        engineering_function (function): The perform_feature_engineering function to be tested.
        response_column (str): The name of the target column to predict.
    
    Returns:
        None
    """
    try:
        engineering_function(response_column)
        logging.info("test_perform_feature_engineering: SUCCESS - Feature engineering applied successfully.")
    except KeyError as err:
        logging.error("test_perform_feature_engineering: FAILURE - Target column does not exist in the dataframe.")
        raise err

    try:
        assert isinstance(response_column, str)
    except AssertionError as err:
        logging.error("test_perform_feature_engineering: FAILURE - Target should be a string.")
        raise err


def test_train_models(training_function):
    """
    Test the train_models function from churn_library.py

    Args:
        training_function (function): The train_models function to be tested.
    
    Returns:
        None
    """
    try:
        training_function()
        logging.info("test_train_models: SUCCESS - Models trained successfully.")
    except Exception as err:
        logging.error(f"test_train_models: FAILURE - Error during model training: {err}")
        raise err


if __name__ == "__main__":
    model_instance = cls.CustomerChurnModel()

    test_import(model_instance.import_data)
    test_eda(model_instance.perform_eda)
    test_encoder_helper(model_instance.encode_categorical_features, [
                        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
    # Correctly passing the response_column argument
    test_perform_feature_engineering(model_instance.perform_feature_engineering, 'Churn')
    test_train_models(model_instance.train_and_evaluate_models)
