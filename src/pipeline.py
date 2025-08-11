from src.logger import logger
from src.exceptions import SandyieException
from src.data_pipeline import run_data_pipeline
from src.model_training import run_model_training_pipeline
from src.model_evaluation import run_model_evaluation_pipeline
from src.model_management import run_model_management_pipeline

def run_ml_pipeline(target_column: str = 'target') -> None:
    """
    Orchestrates the entire MLOps pipeline.
    """
    logger.info("--- Starting Full MLOps Pipeline Execution ---")
    try:
        # 1. Data Ingestion and Splitting
        logger.info("Stage 1: Running Data Pipeline...")
        run_data_pipeline(target_column=target_column)
        logger.info("Stage 1: Data Pipeline Completed.")

        # 2. Model Training
        logger.info("Stage 2: Running Model Training Pipeline...")
        trained_models = run_model_training_pipeline(target_column=target_column)
        logger.info("Stage 2: Model Training Pipeline Completed.")

        # 3. Model Evaluation
        logger.info("Stage 3: Running Model Evaluation Pipeline...")
        evaluated_results = run_model_evaluation_pipeline(trained_models, target_column=target_column)
        logger.info("Stage 3: Model Evaluation Pipeline Completed.")

        # 4. Model Management (Comparison and Dumping)
        logger.info("Stage 4: Running Model Management Pipeline...")
        run_model_management_pipeline(trained_models, evaluated_results)
        logger.info("Stage 4: Model Management Pipeline Completed.")

        logger.info("--- Full MLOps Pipeline Execution Completed Successfully ---")

    except SandyieException as e:
        logger.critical(f"MLOps Pipeline failed due to a custom exception: {e}")
        # Optionally, you can add alerts or notifications here for production systems
    except Exception as e:
        logger.critical(f"MLOps Pipeline failed due to an unexpected error: {e}", exc_info=True)
        # exc_info=True logs the full traceback

if __name__ == "__main__":
    # When running this script, it will execute the entire pipeline.
    # Ensure your 'Train_data.csv' is in 'data/raw/' before running.
    # The 'target_column' should match the actual target column in your CSV.
    run_ml_pipeline(target_column='target') # Adjust 'target' if your column name is different