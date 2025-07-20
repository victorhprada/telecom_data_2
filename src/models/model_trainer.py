import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
import joblib
import sys
import os
import json
from datetime import datetime

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, print_metrics

# Configurar logger
logger = setup_logger('model_trainer', 'logs/model_trainer.log')

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self):
        """Treina modelo de Regressão Logística."""
        logger.info("Treinando Regressão Logística...")
        
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        cv_scores = cross_validate(
            model, self.X_train, self.y_train,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model
        
        # Análise dos coeficientes
        coef_analysis = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        return model, cv_scores, coef_analysis
    
    def train_random_forest(self):
        """Treina modelo Random Forest."""
        logger.info("Treinando Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE
        )
        
        cv_scores = cross_validate(
            model, self.X_train, self.y_train,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        
        # Análise de importância das features
        importance_analysis = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, cv_scores, importance_analysis
    
    def evaluate_model(self, model, model_name):
        """Avalia o modelo nos dados de teste."""
        logger.info(f"Avaliando modelo {model_name}...")
        
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        metrics = calculate_metrics(self.y_test, y_pred, y_prob)
        self.results[model_name] = metrics
        
        return metrics
    
    def save_model(self, model, model_name, metrics, cv_scores, extra_info=None):
        """Salva o modelo e seus resultados."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelo
        model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        
        # Preparar resultados
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in metrics.items()
            },
            'cv_scores': {
                'mean': {
                    k.replace('test_', ''): float(v.mean())
                    for k, v in cv_scores.items() if k.startswith('test_')
                },
                'std': {
                    k.replace('test_', ''): float(v.std())
                    for k, v in cv_scores.items() if k.startswith('test_')
                }
            }
        }
        
        if extra_info is not None:
            results['additional_analysis'] = extra_info
        
        # Salvar resultados
        results_path = os.path.join(REPORTS_DIR, f"{model_name}_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return model_path, results_path
    
    def train_and_evaluate_all(self):
        """Treina e avalia todos os modelos."""
        logger.info("Iniciando treinamento e avaliação de todos os modelos...")
        
        # Treinar e avaliar Regressão Logística
        lr_model, lr_cv, lr_coef = self.train_logistic_regression()
        lr_metrics = self.evaluate_model(lr_model, "LogisticRegression")
        lr_paths = self.save_model(
            lr_model, "LogisticRegression", lr_metrics, lr_cv,
            {'coefficients': lr_coef.to_dict('records')}
        )
        
        # Treinar e avaliar Random Forest
        rf_model, rf_cv, rf_imp = self.train_random_forest()
        rf_metrics = self.evaluate_model(rf_model, "RandomForest")
        rf_paths = self.save_model(
            rf_model, "RandomForest", rf_metrics, rf_cv,
            {'feature_importance': rf_imp.to_dict('records')}
        )
        
        return {
            'LogisticRegression': {
                'metrics': lr_metrics,
                'cv_scores': lr_cv,
                'paths': lr_paths,
                'analysis': lr_coef
            },
            'RandomForest': {
                'metrics': rf_metrics,
                'cv_scores': rf_cv,
                'paths': rf_paths,
                'analysis': rf_imp
            }
        }

def main():
    # Carregar dados
    logger.info("Carregando dados...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_transformed.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test_transformed.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'))
    
    # Criar trainer e executar
    trainer = ModelTrainer(X_train, X_test, y_train['Churn'], y_test['Churn'])
    results = trainer.train_and_evaluate_all()
    
    # Imprimir resultados
    print("\nResultados Finais:")
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        print_metrics(data['metrics'])
        print("\nMédia da Validação Cruzada:")
        for metric, values in data['cv_scores'].items():
            if metric.startswith('test_'):
                print(f"{metric[5:]}: {values.mean():.4f} (+/- {values.std() * 2:.4f})")
    
    logger.info("Treinamento e avaliação concluídos!")

if __name__ == "__main__":
    main() 