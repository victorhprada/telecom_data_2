import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import sys
import os

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RAW_DATA_FILE, FIGURES_DIR, REPORTS_DIR

# Carregar os dados
print("Carregando os dados...")
df = pd.read_csv(RAW_DATA_FILE)

# Calcular correlações para variáveis numéricas
numeric_cols = ['tenure', 'Charges.Monthly', 'Charges.Total', 'SeniorCitizen']
correlation_matrix = df[numeric_cols + ['Churn']].corr()

# Criar heatmap de correlações numéricas
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0, fmt='.2f')
plt.title('Correlação entre Variáveis Numéricas e Churn')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'numeric_correlations.png'))
plt.close()

# Analisar relação entre variáveis categóricas e Churn
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']

# Calcular Cramer's V para variáveis categóricas
def cramers_v(var1, var2):
    confusion_matrix = pd.crosstab(var1, var2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# Calcular Cramer's V para cada variável categórica com Churn
cramers_results = {}
for col in categorical_cols:
    cramers_results[col] = cramers_v(df[col], df['Churn'])

# Ordenar resultados do Cramer's V
cramers_sorted = dict(sorted(cramers_results.items(), key=lambda x: x[1], reverse=True))

# Criar gráfico de barras para Cramer's V
plt.figure(figsize=(12, 6))
plt.bar(cramers_sorted.keys(), cramers_sorted.values())
plt.xticks(rotation=45, ha='right')
plt.title('Força da Associação entre Variáveis Categóricas e Churn (Cramer\'s V)')
plt.ylabel('Cramer\'s V')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'categorical_associations.png'))
plt.close()

# Imprimir resultados
print("\nCorrelações com Churn (variáveis numéricas):")
print("-" * 50)
churn_correlations = correlation_matrix['Churn'].sort_values(ascending=False)
for var, corr in churn_correlations.items():
    if var != 'Churn':
        print(f"{var:20} {corr:>10.3f}")

print("\nAssociação com Churn (variáveis categóricas - Cramer's V):")
print("-" * 50)
for var, v in cramers_sorted.items():
    print(f"{var:20} {v:>10.3f}")

# Análise detalhada das variáveis mais importantes
print("\nAnálise detalhada das variáveis mais importantes:")
print("-" * 50)

# Top 3 variáveis numéricas
print("\nTop 3 variáveis numéricas:")
top_numeric = churn_correlations.head(4)[1:4]  # Excluindo Churn
for var, corr in top_numeric.items():
    print(f"\n{var}:")
    print(df.groupby('Churn')[var].describe())

# Top 3 variáveis categóricas
print("\nTop 3 variáveis categóricas:")
top_categorical = dict(list(cramers_sorted.items())[:3])
for var in top_categorical.keys():
    print(f"\n{var}:")
    print(pd.crosstab(df[var], df['Churn'], normalize='columns').round(3) * 100)

# Salvar um resumo em arquivo
summary_file = os.path.join(REPORTS_DIR, 'correlation_summary.txt')
with open(summary_file, 'w') as f:
    f.write("Resumo das Variáveis Mais Importantes para Prever Churn\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Variáveis Numéricas Mais Correlacionadas:\n")
    f.write("-" * 40 + "\n")
    for var, corr in top_numeric.items():
        f.write(f"{var:20} {corr:>10.3f}\n")
    
    f.write("\nVariáveis Categóricas com Maior Associação:\n")
    f.write("-" * 40 + "\n")
    for var, v in list(cramers_sorted.items())[:3]:
        f.write(f"{var:20} {v:>10.3f}\n")

print("\nAnálise completa! Arquivos gerados:")
print(f"1. {os.path.join(FIGURES_DIR, 'numeric_correlations.png')} - Heatmap de correlações numéricas")
print(f"2. {os.path.join(FIGURES_DIR, 'categorical_associations.png')} - Gráfico de força de associação para variáveis categóricas")
print(f"3. {summary_file} - Resumo das variáveis mais importantes") 