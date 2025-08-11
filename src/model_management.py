import joblib
import os
import json
from datetime import datetime
from src.logger import logger
from src.exceptions import SandyieException
from config.constants import (
    MODELS_DIR, CURRENT_BEST_MODEL_PATH, MODEL_ARCHIVE_DIR,
    METADATA_DIR, MODEL_METADATA_FILE
)
import pickle

def load_model(file_path: str):
    """
    Loads a machine learning model from a .pkl file.

    Args:
        file_path (str): The path to the model .pkl file.

    Returns:
        The loaded model object.

    Raises:
        SandyieException: If the model file is not found or cannot be loaded.
    """
    logger.info(f"Attempting to load model from: {file_path}")
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        logger.warning(f"Model file not found at: {file_path}. This might be expected if no previous model exists.")
        return None
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise SandyieException(f"Failed to load model from {file_path}: {e}")

def save_model(model, file_path: str) -> None:
    """
    Saves a machine learning model to a .pkl file.

    Args:
        model: The model object to save.
        file_path (str): The path where the model should be saved.

    Raises:
        SandyieException: If model saving fails.
    """
    logger.info(f"Attempting to save model to: {file_path}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise SandyieException(f"Failed to save model: {e}")

def load_model_metadata() -> list:
    """
    Loads model metadata from the JSON file.

    Returns:
        list: A list of dictionaries, each representing metadata for a model.

    Raises:
        SandyieException: If metadata loading fails.
    """
    logger.info(f"Attempting to load model metadata from: {MODEL_METADATA_FILE}")
    try:
        if not os.path.exists(MODEL_METADATA_FILE):
            logger.warning(f"Model metadata file not found at {MODEL_METADATA_FILE}. Initializing empty metadata.")
            return []
        with open(MODEL_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        logger.info("Model metadata loaded successfully.")
        return metadata
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {MODEL_METADATA_FILE}. File might be corrupted.")
        raise SandyieException(f"Corrupted metadata file: {MODEL_METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        raise SandyieException(f"Failed to load model metadata: {e}")

def save_model_metadata(metadata: list) -> None:
    """
    Saves model metadata to the JSON file.

    Args:
        metadata (list): The list of dictionaries containing model metadata.

    Raises:
        SandyieException: If metadata saving fails.
    """
    logger.info(f"Attempting to save model metadata to: {MODEL_METADATA_FILE}")
    try:
        os.makedirs(METADATA_DIR, exist_ok=True)
        with open(MODEL_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info("Model metadata saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        raise SandyieException(f"Failed to save model metadata: {e}")

def get_current_best_model_accuracy() -> float:
    """
    Retrieves the accuracy of the currently deployed best model from metadata.

    Returns:
        float: The accuracy of the current best model, or -1.0 if no model exists.
    """
    metadata = load_model_metadata()
    # Assuming the last item in the list is the current best model from the last run
    if metadata:
        return metadata[-1].get('accuracy', -1.0)
    logger.info("No current best model found in metadata. Assuming initial accuracy of -1.0.")
    return -1.0

def update_model_metadata(model_name: str, metrics: dict, is_current_best: bool = False) -> None:
    """
    Adds or updates model metadata in the JSON file.

    Args:
        model_name (str): The name of the model.
        metrics (dict): Dictionary of evaluation metrics for the model.
        is_current_best (bool): True if this model is the new current best.
    """
    metadata = load_model_metadata()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"{model_name}_{timestamp}"
    model_path = os.path.join(MODEL_ARCHIVE_DIR, f"{model_id}.pkl")

    new_entry = {
        'model_id': model_id,
        'algorithm_name': model_name,
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'f1_score': metrics.get('f1_score'),
        'roc_auc': metrics.get('roc_auc', None),
        'training_date': timestamp,
        'is_current_best': is_current_best,
        'model_path': model_path # This path is for the archived model
    }

    # If this is the new best model, mark all others as not current best
    if is_current_best:
        for entry in metadata:
            entry['is_current_best'] = False
        # The current_best_model.pkl always points to the best one for deployment
        new_entry['model_path'] = CURRENT_BEST_MODEL_PATH # Update path for the actual deployed model

    metadata.append(new_entry)
    save_model_metadata(metadata)
    logger.info(f"Metadata updated for model '{model_name}'. Is current best: {is_current_best}")

def run_model_management_pipeline(trained_models, evaluated_results):
    """
    Compares models, saves the best one, and archives all models with metadata.
    """
    logger.info("--- Starting Model Management Pipeline ---")

    try:
        # Determine the best model from the current training run
        best_model_name = max(evaluated_results, key=lambda name: evaluated_results[name]['accuracy'])
        best_model_accuracy = evaluated_results[best_model_name]['accuracy']
        best_model = trained_models[best_model_name]

        logger.info(f"Best model from current run: {best_model_name} with accuracy: {best_model_accuracy:.4f}")

        # Define file paths
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        metadata_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'metadata')
        model_archive_dir = os.path.join(models_dir, 'archive')
        current_best_model_path = os.path.join(models_dir, 'current_best_model.pkl')
        metadata_path = os.path.join(metadata_dir, 'model_metadata.json')

        # Create directories if they don't exist
        os.makedirs(model_archive_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        # Load the model metadata history
        try:
            with open(metadata_path, 'r') as f:
                model_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            model_history = []
            logger.info("No model metadata file found. Creating a new one.")
            
        # Determine the current best model's accuracy from history
        current_best_accuracy = -1.0
        if model_history:
            # Assuming the last item in the list is the current best
            current_best_accuracy = model_history[-1].get('accuracy', -1.0)
            
        logger.info(f"No current best model found in metadata. Assuming initial accuracy of -1.0." if current_best_accuracy == -1.0 else f"Current best model accuracy: {current_best_accuracy:.4f}")

        # Check if the new model is better
        if best_model_accuracy > current_best_accuracy:
            logger.info(f"New model '{best_model_name}' (Accuracy: {best_model_accuracy:.4f}) is better than current best.")
            
            # Create a timestamped name for the archived model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(model_archive_dir, f"{best_model_name}_{timestamp}_{best_model_accuracy:.4f}.pkl")

            # Save the new model to the archive
            logger.info(f"Attempting to save model to: {archive_path}")
            with open(archive_path, 'wb') as file:
                pickle.dump(best_model, file)
            logger.info(f"Model saved successfully to {archive_path}")

            # Overwrite the 'current_best_model.pkl' with the new best model
            logger.info(f"Overwriting old current best model at {current_best_model_path}")
            with open(current_best_model_path, 'wb') as file:
                pickle.dump(best_model, file)
            logger.info(f"Model saved successfully to {current_best_model_path}")

            # --- THIS IS THE CORRECTED LOGIC ---
            new_model_info = {
                'model_name': best_model_name,
                'accuracy': best_model_accuracy,
                'path': current_best_model_path,
                'timestamp': datetime.now().isoformat()
            }
            # Append the new model's info to the list
            model_history.append(new_model_info)
            
            # Save the updated list back to the metadata file
            with open(metadata_path, 'w') as f:
                json.dump(model_history, f, indent=4)
            logger.info("Model metadata updated successfully.")

        else:
            logger.info("New model is not an improvement. Not saving.")
            
        logger.info("--- Model Management Pipeline Completed Successfully ---")

    except Exception as e:
        logger.error(f"Error during model comparison and dumping: {e}")
        raise SandyieException(f"Failed during model comparison and dumping: {e}")

if __name__ == "__main__":
    # This block is for testing the model management pipeline independently
    # You would typically have trained models and evaluation results from previous steps.

    # Dummy setup for testing:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=10, random_state=42)

    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_dummy, y_dummy)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_dummy, y_dummy)

    dummy_trained_models = {
        "LogisticRegression": lr_model,
        "RandomForestClassifier": rf_model
    }

    dummy_evaluated_results = {
        "LogisticRegression": {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1_score': 0.84},
        "RandomForestClassifier": {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.88, 'f1_score': 0.87}
    }

    # Ensure directories exist for testing
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_ARCHIVE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Initialize metadata file if it doesn't exist for a clean test run
    if not os.path.exists(MODEL_METADATA_FILE):
        with open(MODEL_METADATA_FILE, 'w') as f:
            json.dump([], f)

    run_model_management_pipeline(dummy_trained_models, dummy_evaluated_results)

    # Verify current best model
    loaded_best_model = load_model(CURRENT_BEST_MODEL_PATH)
    if loaded_best_model:
        logger.info(f"Loaded current best model: {type(loaded_best_model).__name__}")
    else:
        logger.info("No current best model found after management pipeline.")

    # Verify metadata
    all_metadata = load_model_metadata()
    logger.info(f"All model metadata entries: {len(all_metadata)}")
    for entry in all_metadata:
        if entry.get('is_current_best'):
            logger.info(f"Current best from metadata: {entry.get('algorithm_name')} (Accuracy: {entry.get('accuracy')})")
