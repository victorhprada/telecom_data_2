import os
from pathlib import Path

# Definir o diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.absolute()

# Definir caminhos para diferentes diretórios
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR,
                 MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configurações do projeto
PROJECT_NAME = 'telecom_churn_analysis'
RANDOM_STATE = 42

# Configurações de dados
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'telecom_data.csv')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')

# Configurações de features
NUMERIC_FEATURES = [
    'tenure',
    'Charges.Monthly',
    'Charges.Total',
    'SeniorCitizen'
]

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

TARGET_VARIABLE = 'Churn'

# Configurações de modelagem
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2 