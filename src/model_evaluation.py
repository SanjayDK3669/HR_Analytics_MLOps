import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from src.logger import logger
from src.exceptions import SandyieException
from config.constants import TEST_DATA_PATH

def load_evaluation_data(file_path: str, target_column: str = 'target') -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads evaluation (test) data from a CSV file and separates features (X) and target (y).

    Args:
        file_path (str): Path to the test data CSV.
        target_column (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).

    Raises:
        SandyieException: If data loading fails.
    """
    logger.info(f"Loading evaluation data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            raise SandyieException(f"Target column '{target_column}' not found in evaluation data.")
        X_test = df.drop(columns=[target_column])
        y_test = df[target_column]
        logger.info(f"Evaluation data loaded. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found at: {file_path}")
        raise SandyieException(f"Evaluation data file not found at: {file_path}")
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        raise SandyieException(f"Failed to load evaluation data: {e}")

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates a given model and returns performance metrics.

    Args:
        model: The trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        dict: A dictionary containing evaluation metrics.

    Raises:
        SandyieException: If model evaluation fails.
    """
    model_name = type(model).__name__
    logger.info(f"Evaluating model: {model_name}")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Add ROC AUC if the model supports predict_proba
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                # For binary classification, roc_auc_score expects probabilities of the positive class
                if y_proba.shape[1] > 1:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else: # For models that output a single probability column for binary classification
                    roc_auc = roc_auc_score(y_test, y_proba)
                metrics['roc_auc'] = roc_auc
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC for {model_name}: {e}")
        else:
            logger.warning(f"Model {model_name} does not support predict_proba for ROC AUC calculation.")

        logger.info(f"Evaluation results for {model_name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        raise SandyieException(f"Failed to evaluate model {model_name}: {e}")

def run_model_evaluation_pipeline(trained_models: dict, target_column: str = 'target') -> dict:
    """
    Executes the full model evaluation pipeline for all trained models.

    Args:
        trained_models (dict): A dictionary of trained models.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary where keys are model names and values are their evaluation metrics.
    """
    logger.info("--- Starting Model Evaluation Pipeline ---")
    evaluated_results = {}
    try:
        X_test, y_test = load_evaluation_data(TEST_DATA_PATH, target_column=target_column)

        for name, model in trained_models.items():
            metrics = evaluate_model(model, X_test, y_test)
            evaluated_results[name] = metrics
        logger.info("--- Model Evaluation Pipeline Completed Successfully ---")
        return evaluated_results
    except SandyieException as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in model evaluation pipeline: {e}")
        raise SandyieException(f"Unexpected error in model evaluation pipeline: {e}")

if __name__ == "__main__":
    # This block is for testing the model evaluation pipeline independently
    # Ensure test.csv exists in data/processed/ and you have a dummy model.
    # You might need to run data_pipeline.py and model_training.py first.

    # Dummy model for testing if no actual model is trained yet
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=10, random_state=42)
    dummy_model = LogisticRegression(random_state=42)
    dummy_model.fit(X_dummy, y_dummy) # Train on dummy data

    dummy_trained_models = {"DummyLogisticRegression": dummy_model}

    # Ensure test data exists (run data_pipeline.py first or create dummy)
    if not os.path.exists(TEST_DATA_PATH):
        logger.warning(f"Test data not found at {TEST_DATA_PATH}. Creating dummy test data for evaluation test.")
        X_test_dummy, y_test_dummy = make_classification(n_samples=50, n_features=10, random_state=43)
        dummy_test_df = pd.DataFrame(X_test_dummy, columns=[f'feature_{i}' for i in range(10)])
        dummy_test_df['target'] = y_test_dummy
        os.makedirs(os.path.dirname(TEST_DATA_PATH), exist_ok=True)
        dummy_test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info("Dummy test data created.")

    results = run_model_evaluation_pipeline(dummy_trained_models, target_column='target')
    for model_name, metrics in results.items():
        logger.info(f"Test results for {model_name}: {metrics}")