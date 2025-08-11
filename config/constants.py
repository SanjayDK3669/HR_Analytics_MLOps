
import os

# Base directory for the project (relative to where the script is run, usually /app in Docker)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# IMPORTANT: Place your 'Train_data.csv' in mlops_project/data/raw/ before building the Docker image.
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'Train_data.csv')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'validation.csv')

# --- Model Paths ---
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CURRENT_BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'current_best_model.pkl')
MODEL_ARCHIVE_DIR = os.path.join(MODELS_DIR, 'archive')

# --- Metadata Path ---
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
MODEL_METADATA_FILE = os.path.join(METADATA_DIR, 'model_metadata.json')

# --- Logging Path ---
LOG_FILE_PATH = os.path.join(BASE_DIR, 'logs', 'mlops_pipeline.log')

# --- Data Splitting Ratios ---
TEST_SIZE = 0.18
VALIDATION_SIZE = 0.2 
RANDOM_STATE = 42

# --- Model Parameters (Example - adjust as needed for your data) ---
# For Logistic Regression
LR_PARAMS = {'solver': 'liblinear', 'random_state': RANDOM_STATE}
# For RandomForestClassifier
RF_PARAMS = {'n_estimators': 200, 'random_state': RANDOM_STATE, "criterion" : "entropy", "max_depth": 20}
# For GradientBoostingClassifier
GB_PARAMS = {'n_estimators': 200, 'learning_rate': 0.04, 'random_state': RANDOM_STATE}
# For SVC
SVC_PARAMS = {'kernel': 'linear', 'random_state': RANDOM_STATE, 'probability': True}

# --- Frontend Configuration ---
FLASK_PORT = 5000