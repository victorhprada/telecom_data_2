from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """Calcula métricas de avaliação para modelos de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo (classes)
        y_prob: Probabilidades preditas (opcional)
    
    Returns:
        Dicionário com as métricas calculadas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics

def print_metrics(metrics: Dict[str, Any]) -> None:
    """Imprime métricas de forma formatada.
    
    Args:
        metrics: Dicionário com métricas calculadas
    """
    print("\nMétricas de Avaliação:")
    print("-" * 30)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nMatriz de Confusão:")
    print(metrics['confusion_matrix']) 