import pandas as pd
import numpy as np

df = pd.read_csv('telecom_data.csv')

# Análise inicial do dataset
print("Análise inicial do dataset:")
print("-" * 50)
print(f"Shape original: {df.shape}")

# Identificar colunas com valores únicos
unique_counts = df.nunique()
print("\nColunas com alto número de valores únicos (possíveis identificadores):")
print(unique_counts[unique_counts > df.shape[0] * 0.5])

# Identificar colunas numéricas para correlação
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_columns].corr()

# Identificar pares de colunas altamente correlacionadas
print("\nPares de colunas altamente correlacionadas (|corr| > 0.9):")
high_corr = np.where(np.abs(correlation_matrix) > 0.9)
for i, j in zip(*high_corr):
    if i < j:  # Evitar duplicatas
        print(f"{numeric_columns[i]} - {numeric_columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")

# Identificar colunas com versões transformadas
transformed_columns = [col for col in df.columns if any(suffix in col for suffix in ['_scaled', '_log'])]
print("\nColunas com versões transformadas:")
print(transformed_columns)

# Sugestão de colunas para remover
print("\nSugestão de colunas que podem ser removidas:")
columns_to_remove = ['customerID']  # Identificador único
columns_to_remove.extend(transformed_columns)  # Versões transformadas

print("\nColunas sugeridas para remoção:")
for col in columns_to_remove:
    print(f"- {col}")

print(f"\nShape após remoção das colunas sugeridas: {df.shape[0]} x {df.shape[1] - len(columns_to_remove)}")








