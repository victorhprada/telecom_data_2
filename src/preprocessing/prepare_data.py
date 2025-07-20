import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
import os

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (RAW_DATA_FILE, PROCESSED_DATA_DIR, NUMERIC_FEATURES,
                   CATEGORICAL_FEATURES, TEST_SIZE, RANDOM_STATE)

# Carregar os dados
print("Carregando o dataset...")
df = pd.read_csv(RAW_DATA_FILE)
print(f"Shape original: {df.shape}")

# Lista de colunas para remover
colunas_para_remover = [
    # Identificadores
    'customerID',
    
    # Versões escaladas/transformadas
    'SeniorCitizen_log',
    'tenure_scaled',
    'Charges.Monthly_scaled',
    'Charges.Total_scaled',
    'SeniorCitizen_scaled',
    'Contas_Diarias_scaled',
    
    # Colunas redundantes
    'Contas_Diarias',  # Correlacionada com Charges.Monthly
    
    # Colunas redundantes de serviço de internet
    'OnlineSecurity_No internet service',
    'OnlineBackup_No internet service',
    'DeviceProtection_No internet service',
    'TechSupport_No internet service',
    'StreamingTV_No internet service',
    'StreamingMovies_No internet service'
]

# Remover as colunas
df_clean = df.drop(columns=colunas_para_remover)
print(f"\nShape após remoção de colunas: {df_clean.shape}")

# Criar preprocessador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

# Separar features e target
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Aplicar transformações
print("\nAplicando transformações...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Obter nomes das features após one-hot encoding
cat_feature_names = []
for i, encoder in enumerate(preprocessor.named_transformers_['cat'].categories_):
    cat_feature_names.extend([f"{CATEGORICAL_FEATURES[i]}_{val}" for val in encoder[1:]])

feature_names = NUMERIC_FEATURES + cat_feature_names

# Converter para DataFrame
X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

print("\nDimensões dos conjuntos de dados transformados:")
print(f"X_train: {X_train_transformed.shape}")
print(f"X_test: {X_test_transformed.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Salvar os dados preparados
print("\nSalvando os dados preparados...")
X_train_transformed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_transformed.csv'), index=False)
X_test_transformed.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_transformed.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)

# Mostrar informações sobre as features finais
print("\nFeatures finais após transformação:")
print("\nFeatures numéricas (padronizadas):")
for col in NUMERIC_FEATURES:
    print(f"- {col}")

print("\nFeatures categóricas (one-hot encoded, primeira categoria como referência):")
for col in CATEGORICAL_FEATURES:
    print(f"- {col}")

print("\nPreparação dos dados concluída!")