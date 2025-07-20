import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import joblib
import sys
import os
import json
from datetime import datetime

# Adicionar o diretório raiz ao path para importar config e utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, print_metrics

# Configurar logger
logger = setup_logger('train_models', 'logs/train_models.log')

def load_data():
    """Carregar dados preprocessados."""
    logger.info("Carregando dados de treino e teste...")
    
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_transformed.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_transformed.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'))
    
    return X_train, X_test, y_train['Churn'], y_test['Churn']

def train_logistic_regression(X_train, y_train):
    """Treinar modelo de Regressão Logística."""
    logger.info("Treinando modelo de Regressão Logística...")
    
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    
    # Realizar validação cruzada
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    # Treinar modelo final
    model.fit(X_train, y_train)
    
    return model, cv_scores

def train_random_forest(X_train, y_train):
    """Treinar modelo Random Forest."""
    logger.info("Treinando modelo Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    
    # Realizar validação cruzada
    cv_scores = cross_validate(
        model, X_train, y_train,
        cv=5,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    # Treinar modelo final
    model.fit(X_train, y_train)
    
    return model, cv_scores

def evaluate_model(model, X_test, y_test, model_name):
    """Avaliar modelo nos dados de teste."""
    logger.info(f"Avaliando modelo {model_name}...")
    
    # Fazer predições
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Converter arrays numpy para listas
    metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    # Se for Random Forest, calcular importância das features
    if isinstance(model, RandomForestClassifier):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = {
            'features': feature_importance['feature'].tolist(),
            'importance': feature_importance['importance'].tolist()
        }
    
    return metrics

def save_results(model, metrics, cv_scores, model_name):
    """Salvar modelo e resultados."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    
    # Preparar resultados
    results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        'cv_scores': {
            'accuracy': float(cv_scores['test_accuracy'].mean()),
            'precision': float(cv_scores['test_precision'].mean()),
            'recall': float(cv_scores['test_recall'].mean()),
            'f1': float(cv_scores['test_f1'].mean()),
            'roc_auc': float(cv_scores['test_roc_auc'].mean()),
            'std': {
                'accuracy': float(cv_scores['test_accuracy'].std()),
                'precision': float(cv_scores['test_precision'].std()),
                'recall': float(cv_scores['test_recall'].std()),
                'f1': float(cv_scores['test_f1'].std()),
                'roc_auc': float(cv_scores['test_roc_auc'].std())
            }
        }
    }
    
    # Salvar resultados
    results_path = os.path.join(REPORTS_DIR, f"{model_name}_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return model_path, results_path

def main():
    """Função principal para treinar e avaliar modelos."""
    logger.info("Iniciando treinamento dos modelos...")
    
    # Carregar dados
    X_train, X_test, y_train, y_test = load_data()
    
    # Treinar e avaliar Regressão Logística
    lr_model, lr_cv_scores = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
    lr_model_path, lr_results_path = save_results(lr_model, lr_metrics, lr_cv_scores, "LogisticRegression")
    
    # Treinar e avaliar Random Forest
    rf_model, rf_cv_scores = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    rf_model_path, rf_results_path = save_results(rf_model, rf_metrics, rf_cv_scores, "RandomForest")
    
    # Imprimir resultados comparativos
    print("\nResultados da Regressão Logística:")
    print("-" * 40)
    print_metrics(lr_metrics)
    print("\nMédia da Validação Cruzada:")
    for metric, value in lr_cv_scores.items():
        if metric.startswith('test_'):
            print(f"{metric[5:]}: {value.mean():.4f} (+/- {value.std() * 2:.4f})")
    
    print("\nResultados do Random Forest:")
    print("-" * 40)
    print_metrics(rf_metrics)
    print("\nMédia da Validação Cruzada:")
    for metric, value in rf_cv_scores.items():
        if metric.startswith('test_'):
            print(f"{metric[5:]}: {value.mean():.4f} (+/- {value.std() * 2:.4f})")
    
    # Imprimir importância das features (Random Forest)
    if 'feature_importance' in rf_metrics:
        print("\nTop 10 Features Mais Importantes (Random Forest):")
        features = rf_metrics['feature_importance']['features'][:10]
        importance = rf_metrics['feature_importance']['importance'][:10]
        for f, i in zip(features, importance):
            print(f"{f}: {i:.4f}")
    
    logger.info("Treinamento e avaliação dos modelos concluídos!")
    logger.info(f"Modelos salvos em: {MODELS_DIR}")
    logger.info(f"Resultados salvos em: {REPORTS_DIR}")

if __name__ == "__main__":
    main() 