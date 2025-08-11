import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exceptions import SandyieException
from config.constants import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    VALIDATION_DATA_PATH, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
)

def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Ingests data from a specified CSV file path.

    Args:
        file_path (str): The path to the CSV data file.

    Returns:
        pd.DataFrame: The ingested data.

    Raises:
        SandyieException: If the file is not found or cannot be read.
    """
    logger.info(f"Attempting to ingest data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully ingested data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at: {file_path}")
        raise SandyieException(f"Data file not found at: {file_path}")
    except Exception as e:
        logger.error(f"Error ingesting data from {file_path}: {e}")
        raise SandyieException(f"Failed to ingest data: {e}")

def split_data(df: pd.DataFrame, target_column: str = 'target') -> None:
    """
    Splits the input DataFrame into training, testing, and validation sets,
    and saves them to CSV files.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        target_column (str): The name of the target column.

    Raises:
        SandyieException: If data splitting fails.
    """
    logger.info("Starting data splitting process.")
    try:
        # Create processed data directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # First split: 80% train_val, 20% test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"Initial split: Train+Validation shape {X_train_val.shape}, Test shape {X_test.shape}")

        # Second split: From train_val, 87.5% train, 12.5% validation (0.125 / 0.8 = 0.15625)
        # This results in: 70% train, 10% validation, 20% test (approx)
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_train_val
        )
        logger.info(f"Second split: Train shape {X_train.shape}, Validation shape {X_val.shape}")

        # Combine X and y for saving
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)

        # Save to CSV
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        val_df.to_csv(VALIDATION_DATA_PATH, index=False)

        logger.info(f"Data split and saved to:\nTrain: {TRAIN_DATA_PATH}\nTest: {TEST_DATA_PATH}\nValidation: {VALIDATION_DATA_PATH}")

    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise SandyieException(f"Failed to split data: {e}")

def run_data_pipeline(target_column: str = 'target'):
    """
    Executes the full data ingestion and splitting pipeline.
    """
    logger.info("--- Starting Data Pipeline ---")
    try:
        df = ingest_data(RAW_DATA_PATH)
        # Assuming the target column is named 'target'. Adjust if your data has a different name.
        # If your data doesn't have a 'target' column, you might need to
        # generate a dummy one for demonstration or adjust this function.
        if target_column not in df.columns:
            logger.warning(f"Target column '{target_column}' not found. Generating a dummy target column for demonstration.")
            # For demonstration, create a dummy binary target column if not present
            df[target_column] = (df.iloc[:, 0] > df.iloc[:, 0].median()).astype(int)
            logger.info(f"Dummy target column '{target_column}' created.")

        split_data(df, target_column=target_column)
        logger.info("--- Data Pipeline Completed Successfully ---")
    except SandyieException as e:
        logger.error(f"Data pipeline failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in data pipeline: {e}")
        raise SandyieException(f"Unexpected error in data pipeline: {e}")

if __name__ == "__main__":
    # This block is for testing the data pipeline independently
    # For a real run, ensure Train_data.csv exists in data/raw/
    # You might need to create a dummy CSV for local testing if you don't have it yet.
    # Example dummy data creation:
    if not os.path.exists(RAW_DATA_PATH):
        logger.warning(f"Raw data file not found at {RAW_DATA_PATH}. Creating dummy data for testing.")
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(
            n_samples=1000, n_features=10, n_informative=5, n_redundant=0,
            n_classes=2, random_state=RANDOM_STATE
        )
        dummy_df = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(10)])
        dummy_df['target'] = y_dummy
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        dummy_df.to_csv(RAW_DATA_PATH, index=False)
        logger.info("Dummy data created at data/raw/Train_data.csv")

    run_data_pipeline(target_column='target') # Adjust target_column if different