import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import os
from src.logger import logger
from src.exceptions import SandyieException
from config.constants import (
    TRAIN_DATA_PATH, LR_PARAMS, RF_PARAMS, GB_PARAMS, SVC_PARAMS, MODELS_DIR
)

def load_training_data(file_path: str, target_column: str = 'target') -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads training data from a CSV file and separates features (X) and target (y).

    Args:
        file_path (str): Path to the training data CSV.
        target_column (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).

    Raises:
        SandyieException: If data loading fails.
    """
    logger.info(f"Loading training data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            raise SandyieException(f"Target column '{target_column}' not found in training data.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info(f"Training data loaded. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    except FileNotFoundError:
        logger.error(f"Training data file not found at: {file_path}")
        raise SandyieException(f"Training data file not found at: {file_path}")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise SandyieException(f"Failed to load training data: {e}")

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Trains multiple machine learning models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        dict: A dictionary containing trained models, with algorithm names as keys.

    Raises:
        SandyieException: If any model training fails.
    """
    logger.info("Starting model training with multiple algorithms.")
    trained_models = {}

    algorithms = {
        "LogisticRegression": (LogisticRegression, LR_PARAMS),
        "RandomForestClassifier": (RandomForestClassifier, RF_PARAMS),
        "GradientBoostingClassifier": (GradientBoostingClassifier, GB_PARAMS),
        #"SVC": (SVC, SVC_PARAMS) # Note: SVC can be slow on large datasets
    }

    for name, (model_class, params) in algorithms.items():
        try:
            logger.info(f"Training {name}...")
            model = model_class(**params)
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"{name} training completed.")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            raise SandyieException(f"Failed to train {name}: {e}")

    return trained_models

def run_model_training_pipeline(target_column: str = 'target') -> dict:
    """
    Executes the full model training pipeline.

    Args:
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary of trained models.
    """
    logger.info("--- Starting Model Training Pipeline ---")
    try:
        X_train, y_train = load_training_data(TRAIN_DATA_PATH, target_column=target_column)
        trained_models = train_models(X_train, y_train)
        logger.info("--- Model Training Pipeline Completed Successfully ---")
        return trained_models
    except SandyieException as e:
        logger.error(f"Model training pipeline failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in model training pipeline: {e}")
        raise SandyieException(f"Unexpected error in model training pipeline: {e}")

if __name__ == "__main__":
    # This block is for testing the model training pipeline independently
    # Ensure train.csv exists in data/processed/
    # You might need to run data_pipeline.py first to generate the data.
    if not os.path.exists(TRAIN_DATA_PATH):
        logger.error(f"Training data not found at {TRAIN_DATA_PATH}. Please run data_pipeline.py first.")
        # Exit or create dummy data for testing
        raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}")

    trained_models = run_model_training_pipeline(target_column='target')
    for name, model in trained_models.items():
        logger.info(f"Trained model: {name}, type: {type(model)}")