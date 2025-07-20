import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RAW_DATA_FILE, FIGURES_DIR, REPORTS_DIR

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
sns.set_palette("husl")

# Carregar os dados
print("Carregando os dados...")
df = pd.read_csv(RAW_DATA_FILE)

# 1. Análise de Contrato vs Churn
print("\nAnalisando relação entre tipo de contrato e churn...")

# Calcular taxa de churn por tipo de contrato
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print("\nTaxa de Churn por Tipo de Contrato:")
print(contract_churn)

# Visualizar taxa de churn por tipo de contrato
plt.figure(figsize=(10, 6))
contract_churn[True].plot(kind='bar')
plt.title('Taxa de Churn por Tipo de Contrato')
plt.xlabel('Tipo de Contrato')
plt.ylabel('Taxa de Churn (%)')
plt.xticks(rotation=45)
for i, v in enumerate(contract_churn[True]):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'contract_churn_rate.png'))
plt.close()

# 2. Análise de Tempo como Cliente (tenure) vs Churn
print("\nAnalisando relação entre tempo como cliente e churn...")

# Estatísticas de tenure por status de churn
tenure_stats = df.groupby('Churn')['tenure'].describe()
print("\nEstatísticas de Tempo como Cliente por Status de Churn:")
print(tenure_stats)

# Visualizar distribuição de tenure por churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Contract', y='tenure', hue='Churn', data=df)
plt.title('Distribuição do Tempo como Cliente por Contrato e Churn')
plt.xlabel('Tipo de Contrato')
plt.ylabel('Tempo como Cliente (meses)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'tenure_contract_churn.png'))
plt.close()

# 3. Análise de Gastos vs Churn
print("\nAnalisando relação entre gastos e churn...")

# Estatísticas de gastos mensais por status de churn
charges_stats = df.groupby('Churn')['Charges.Monthly'].describe()
print("\nEstatísticas de Gastos Mensais por Status de Churn:")
print(charges_stats)

# Visualizar gastos mensais por tipo de contrato e churn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Contract', y='Charges.Monthly', hue='Churn', data=df)
plt.title('Distribuição dos Gastos Mensais por Contrato e Churn')
plt.xlabel('Tipo de Contrato')
plt.ylabel('Gastos Mensais ($)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'charges_contract_churn.png'))
plt.close()

# 4. Análise Combinada de Tempo, Gastos e Churn
print("\nAnalisando relação combinada entre tempo, gastos e churn...")

# Criar gráfico de dispersão
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='tenure', y='Charges.Monthly', hue='Churn', 
                style='Contract', s=100, alpha=0.6)
plt.title('Relação entre Tempo como Cliente, Gastos Mensais e Churn')
plt.xlabel('Tempo como Cliente (meses)')
plt.ylabel('Gastos Mensais ($)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'tenure_charges_churn.png'))
plt.close()

# 5. Salvar resumo das análises
print("\nSalvando resumo das análises...")

with open(os.path.join(REPORTS_DIR, 'contract_charges_analysis.txt'), 'w') as f:
    f.write("Análise de Contratos, Tempo como Cliente e Gastos vs Churn\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("1. Taxa de Churn por Tipo de Contrato\n")
    f.write("-" * 40 + "\n")
    f.write(contract_churn.to_string())
    f.write("\n\n")
    
    f.write("2. Estatísticas de Tempo como Cliente por Status de Churn\n")
    f.write("-" * 40 + "\n")
    f.write(tenure_stats.to_string())
    f.write("\n\n")
    
    f.write("3. Estatísticas de Gastos Mensais por Status de Churn\n")
    f.write("-" * 40 + "\n")
    f.write(charges_stats.to_string())
    f.write("\n\n")
    
    # Adicionar correlações
    f.write("4. Correlações\n")
    f.write("-" * 40 + "\n")
    correlations = df[['tenure', 'Charges.Monthly', 'Churn']].corr()
    f.write(correlations.to_string())

print("\nAnálise completa! Arquivos gerados:")
print(f"1. {os.path.join(FIGURES_DIR, 'contract_churn_rate.png')} - Taxa de Churn por Tipo de Contrato")
print(f"2. {os.path.join(FIGURES_DIR, 'tenure_contract_churn.png')} - Distribuição do Tempo como Cliente")
print(f"3. {os.path.join(FIGURES_DIR, 'charges_contract_churn.png')} - Distribuição dos Gastos Mensais")
print(f"4. {os.path.join(FIGURES_DIR, 'tenure_charges_churn.png')} - Relação entre Tempo, Gastos e Churn")
print(f"5. {os.path.join(REPORTS_DIR, 'contract_charges_analysis.txt')} - Resumo das análises") 