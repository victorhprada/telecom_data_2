import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados balanceados (usando SMOTETomek como exemplo)
print("Carregando os dados...")
X_train = pd.read_csv('data/X_train_smotetomek.csv')
X_test = pd.read_csv('data/X_test_transformed.csv')

# Analisar estatísticas das features numéricas
print("\nEstatísticas descritivas das features antes da normalização:")
print(X_train.describe())

# Identificar features numéricas
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
print(f"\nFeatures numéricas identificadas: {len(numeric_features)}")
for feat in numeric_features:
    print(f"- {feat}")

# Criar visualização da distribuição das features numéricas
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
X_train[numeric_features].boxplot()
plt.title('Distribuição Original das Features Numéricas')
plt.xticks(rotation=45)

# Aplicar StandardScaler
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Visualizar dados após padronização
plt.subplot(1, 2, 2)
X_train_scaled[numeric_features].boxplot()
plt.title('Distribuição após StandardScaler')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_scaling_comparison.png')
plt.close()

# Salvar versões dos dados para diferentes tipos de modelos
print("\nPreparando dados para diferentes tipos de modelos...")

# 1. Para modelos baseados em distância (com scaling)
print("\n1. Salvando dados normalizados para modelos baseados em distância:")
print("   (KNN, SVM, Regressão Logística, Redes Neurais)")
X_train_scaled.to_csv('data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/X_test_scaled.csv', index=False)

# 2. Para modelos baseados em árvore (sem scaling)
print("\n2. Mantendo dados originais para modelos baseados em árvore:")
print("   (Decision Tree, Random Forest, XGBoost)")
X_train.to_csv('data/X_train_tree.csv', index=False)
X_test.to_csv('data/X_test_tree.csv', index=False)

# Análise das escalas
print("\nAnálise das escalas dos dados:")
print("\nAntes da padronização:")
print(X_train[numeric_features].agg(['min', 'max', 'mean', 'std']).round(2))

print("\nApós a padronização:")
print(X_train_scaled[numeric_features].agg(['min', 'max', 'mean', 'std']).round(2))

# Recomendações
print("\nRecomendações para diferentes modelos:")
print("\n1. Para modelos baseados em distância (KNN, SVM, Regressão Logística, Redes Neurais):")
print("   - Usar os dados padronizados: 'X_train_scaled.csv' e 'X_test_scaled.csv'")
print("   - Features numéricas foram padronizadas com média 0 e desvio padrão 1")
print("   - Features categóricas já estão adequadamente codificadas (one-hot encoding)")

print("\n2. Para modelos baseados em árvore (Decision Tree, Random Forest, XGBoost):")
print("   - Usar os dados originais: 'X_train_tree.csv' e 'X_test_tree.csv'")
print("   - Não é necessário padronização")
print("   - Features categóricas já estão adequadamente codificadas (one-hot encoding)")

print("\nObservação: Um gráfico comparativo foi salvo como 'feature_scaling_comparison.png'") 