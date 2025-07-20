import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import sys
import os
from datetime import datetime

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RAW_DATA_FILE, FIGURES_DIR, REPORTS_DIR
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('feature_analysis', 'logs/feature_analysis.log')

class FeatureAnalyzer:
    def __init__(self, data):
        self.df = data
        self.numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns
        
    def analyze_correlations(self):
        """Analisa correlações entre variáveis numéricas."""
        logger.info("Analisando correlações...")
        
        correlation_matrix = self.df[self.numeric_cols].corr()
        
        # Criar heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0)
        plt.title('Correlações entre Variáveis Numéricas')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
        plt.close()
        
        return correlation_matrix
    
    def analyze_categorical_associations(self):
        """Analisa associações entre variáveis categóricas e target (Churn)."""
        logger.info("Analisando associações categóricas...")
        
        associations = {}
        for col in self.categorical_cols:
            if col != 'Churn':
                contingency = pd.crosstab(self.df[col], self.df['Churn'])
                chi2, p_value, _, _ = chi2_contingency(contingency)
                associations[col] = {'chi2': chi2, 'p_value': p_value}
        
        return pd.DataFrame(associations).T.sort_values('chi2', ascending=False)
    
    def analyze_feature_importance_for_churn(self):
        """Analisa importância das features para prever churn."""
        logger.info("Analisando importância das features para churn...")
        
        # Análise numérica
        numeric_analysis = {}
        for col in self.numeric_cols:
            if col != 'Churn':
                stats = self.df.groupby('Churn')[col].describe()
                effect_size = (stats.loc[True, 'mean'] - stats.loc[False, 'mean']) / stats.loc[False, 'std']
                numeric_analysis[col] = {
                    'effect_size': effect_size,
                    'stats': stats
                }
        
        # Análise categórica
        categorical_analysis = {}
        for col in self.categorical_cols:
            if col != 'Churn':
                proportions = pd.crosstab(self.df[col], self.df['Churn'], normalize='columns')
                categorical_analysis[col] = proportions
        
        return numeric_analysis, categorical_analysis
    
    def analyze_redundancy(self):
        """Identifica possíveis redundâncias nos dados."""
        logger.info("Analisando redundâncias...")
        
        # Correlações altas
        corr_matrix = self.df[self.numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        # Features derivadas
        derived_features = []
        for col in self.numeric_cols:
            scaled_versions = [c for c in self.numeric_cols if c.startswith(f"{col}_") or c.endswith("_scaled")]
            if scaled_versions:
                derived_features.append({
                    'original': col,
                    'derived': scaled_versions
                })
        
        return pd.DataFrame(high_corr_pairs), pd.DataFrame(derived_features)
    
    def generate_report(self):
        """Gera relatório completo de análise."""
        logger.info("Gerando relatório completo...")
        
        # Realizar todas as análises
        correlations = self.analyze_correlations()
        cat_associations = self.analyze_categorical_associations()
        num_importance, cat_importance = self.analyze_feature_importance_for_churn()
        high_corr, derived = self.analyze_redundancy()
        
        # Preparar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = []
        
        report.append("Análise Completa de Features")
        report.append("=" * 80)
        report.append(f"\nData da Análise: {timestamp}")
        
        # Seção 1: Correlações
        report.append("\n1. Correlações Significativas")
        report.append("-" * 50)
        for _, row in high_corr.iterrows():
            report.append(f"- {row['feature1']} e {row['feature2']}: {row['correlation']:.3f}")
        
        # Seção 2: Associações Categóricas
        report.append("\n2. Associações com Churn (Chi-Square)")
        report.append("-" * 50)
        for feature, row in cat_associations.iterrows():
            report.append(f"- {feature}: chi2={row['chi2']:.2f}, p={row['p_value']:.4f}")
        
        # Seção 3: Importância para Churn
        report.append("\n3. Efeito das Features Numéricas no Churn")
        report.append("-" * 50)
        for feature, data in num_importance.items():
            report.append(f"- {feature}: effect_size={data['effect_size']:.3f}")
        
        # Seção 4: Redundâncias
        report.append("\n4. Features Potencialmente Redundantes")
        report.append("-" * 50)
        for _, row in derived.iterrows():
            report.append(f"- {row['original']} tem versões: {', '.join(row['derived'])}")
        
        # Salvar relatório
        report_path = os.path.join(REPORTS_DIR, f'feature_analysis_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Relatório salvo em: {report_path}")
        return report_path

def main():
    # Carregar dados
    logger.info("Carregando dados...")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Criar analisador e gerar relatório
    analyzer = FeatureAnalyzer(df)
    report_path = analyzer.generate_report()
    
    print(f"Análise completa! Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main() 