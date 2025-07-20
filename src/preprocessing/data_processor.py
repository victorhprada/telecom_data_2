import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import sys
import os
from datetime import datetime

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (RAW_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_STATE,
                   NUMERIC_FEATURES, CATEGORICAL_FEATURES)
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('data_processor', 'logs/data_processor.log')

class DataProcessor:
    def __init__(self, data=None):
        self.df = data if data is not None else pd.read_csv(RAW_DATA_FILE)
        self.numeric_features = NUMERIC_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        
    def remove_redundant_features(self):
        """Remove features redundantes e derivadas."""
        logger.info("Removendo features redundantes...")
        
        redundant_features = [
            # Identificadores
            'customerID',
            
            # Versões escaladas/transformadas
            'SeniorCitizen_log',
            'tenure_scaled',
            'Charges.Monthly_scaled',
            'Charges.Total_scaled',
            'SeniorCitizen_scaled',
            'Contas_Diarias_scaled',
            
            # Features redundantes
            'Contas_Diarias',
            
            # Colunas redundantes de serviço
            'OnlineSecurity_No internet service',
            'OnlineBackup_No internet service',
            'DeviceProtection_No internet service',
            'TechSupport_No internet service',
            'StreamingTV_No internet service',
            'StreamingMovies_No internet service'
        ]
        
        self.df = self.df.drop(columns=[col for col in redundant_features if col in self.df.columns])
        return self
    
    def handle_missing_values(self):
        """Trata valores ausentes."""
        logger.info("Tratando valores ausentes...")
        
        # Verificar valores ausentes
        missing = self.df.isnull().sum()
        if missing.any():
            logger.warning(f"Encontrados valores ausentes:\n{missing[missing > 0]}")
            self.df = self.df.dropna()
            logger.info(f"Shape após remoção de valores ausentes: {self.df.shape}")
        
        return self
    
    def scale_numeric_features(self):
        """Padroniza features numéricas."""
        logger.info("Padronizando features numéricas...")
        
        scaler = StandardScaler()
        self.df[self.numeric_features] = scaler.fit_transform(self.df[self.numeric_features])
        return self
    
    def split_data(self, test_size=0.2):
        """Divide os dados em treino e teste."""
        logger.info("Dividindo dados em treino e teste...")
        
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def balance_data(self, X, y, method='smote'):
        """Aplica técnicas de balanceamento nos dados."""
        logger.info(f"Aplicando balanceamento usando {method}...")
        
        if method == 'smote':
            balancer = SMOTE(random_state=RANDOM_STATE)
        elif method == 'undersampling':
            balancer = RandomUnderSampler(random_state=RANDOM_STATE)
        elif method == 'smotetomek':
            balancer = SMOTETomek(random_state=RANDOM_STATE)
        else:
            raise ValueError(f"Método de balanceamento '{method}' não suportado")
        
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Salva os dados processados."""
        logger.info("Salvando dados processados...")
        
        # Criar diretório se não existir
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Salvar dados
        X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_transformed.csv'), index=False)
        X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_transformed.csv'), index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
        
        logger.info(f"Dados salvos em: {PROCESSED_DATA_DIR}")
    
    def process_all(self, balance_method=None):
        """Executa todo o pipeline de processamento."""
        logger.info("Iniciando pipeline de processamento...")
        
        # Pré-processamento
        self.remove_redundant_features()
        self.handle_missing_values()
        self.scale_numeric_features()
        
        # Split dos dados
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Balanceamento (opcional)
        if balance_method:
            X_train, y_train = self.balance_data(X_train, y_train, balance_method)
        
        # Salvar dados
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test

def main():
    # Processar dados
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.process_all(balance_method='smotetomek')
    
    # Imprimir informações sobre os dados processados
    print("\nInformações dos dados processados:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    if isinstance(y_train, pd.Series):
        print("\nDistribuição das classes (treino):")
        print(y_train.value_counts(normalize=True))
    
    logger.info("Processamento concluído!")

if __name__ == "__main__":
    main() 