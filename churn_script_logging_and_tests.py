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
        
        # Assertions to ensure the dataframe is not empty
        assert df.shape[0] > 0, "The dataframe has no rows"
        assert df.shape[1] > 0, "The dataframe has no columns"
        
    except FileNotFoundError as err:
        logging.error("test_import: FAILURE - The specified file was not found.")
        raise err
    except AssertionError as err:
        logging.error(f"test_import: FAILURE - {err}")
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
        
        # Verify that the EDA images are generated correctly
        expected_files = [
            './images/eda/Customer_Age_distribution.png',  # Example for a numeric column
            './images/eda/Gender_distribution.png',  # Example for a categorical column
            './images/eda/correlation_heatmap.png'
        ]
        
        for file_path in expected_files:
            assert os.path.exists(file_path), f"{file_path} not found"
        
    except Exception as err:
        logging.error(f"test_eda: FAILURE - Error during EDA: {err}")
        raise err
    except AssertionError as err:
        logging.error(f"test_eda: FAILURE - {err}")
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
        df_encoded = encoder_function(categorical_columns)
        logging.info("test_encoder_helper: SUCCESS - Encoding applied successfully.")
        
        # Assertions to ensure the encoded dataframe is not empty
        assert df_encoded.shape[0] > 0, "The encoded dataframe has no rows"
        assert df_encoded.shape[1] > 0, "The encoded dataframe has no columns"
        for col in categorical_columns:
            assert any(col in s for s in df_encoded.columns), f"Column {col} not encoded properly"
        
    except KeyError as err:
        logging.error("test_encoder_helper: FAILURE - A column specified does not exist in the dataframe.")
        raise err
    except AssertionError as err:
        logging.error(f"test_encoder_helper: FAILURE - {err}")
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
        x_train, x_test, y_train, y_test = engineering_function(response_column)
        logging.info("test_perform_feature_engineering: SUCCESS - Feature engineering applied successfully.")
        
        # Assertions to ensure the split data is not empty
        assert x_train.shape[0] > 0, "x_train has no rows"
        assert x_test.shape[0] > 0, "x_test has no rows"
        assert y_train.shape[0] > 0, "y_train has no rows"
        assert y_test.shape[0] > 0, "y_test has no rows"
        
    except KeyError as err:
        logging.error("test_perform_feature_engineering: FAILURE - Target column does not exist in the dataframe.")
        raise err
    except AssertionError as err:
        logging.error(f"test_perform_feature_engineering: FAILURE - {err}")
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
        
        # Assertions to check that model files and evaluation plots are created
        assert os.path.exists('./models/logistic_model.pkl'), "Logistic model file not found"
        assert os.path.exists('./models/rfc_model.pkl'), "Random Forest model file not found"
        assert os.path.exists('./images/results/roc_curve_result.png'), "ROC curve result not found"
        
    except Exception as err:
        logging.error(f"test_train_models: FAILURE - Error during model training: {err}")
        raise err
    except AssertionError as err:
        logging.error(f"test_train_models: FAILURE - {err}")
        raise err


if __name__ == "__main__":
    model_instance = cls.CustomerChurnModel()

    test_import(model_instance.import_data)
    test_eda(model_instance.perform_eda)
    test_encoder_helper(model_instance.encode_categorical_features, [
                        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
    test_perform_feature_engineering(model_instance.perform_feature_engineering, 'Churn')
    test_train_models(model_instance.train_and_evaluate_models)
