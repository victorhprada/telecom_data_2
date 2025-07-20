import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
print("Carregando o dataset...")
df = pd.read_csv('telecom_data.csv')

# Calcular a distribuição de Churn
churn_distribution = df['Churn'].value_counts()
churn_percentages = df['Churn'].value_counts(normalize=True) * 100

# Criar figura com dois subplots
plt.figure(figsize=(15, 6))

# Subplot 1: Números absolutos
plt.subplot(1, 2, 1)
sns.barplot(x=churn_distribution.index, y=churn_distribution.values)
plt.title('Distribuição de Churn (Números Absolutos)')
plt.xlabel('Churn')
plt.ylabel('Número de Clientes')
for i, v in enumerate(churn_distribution.values):
    plt.text(i, v, str(v), ha='center', va='bottom')

# Subplot 2: Porcentagens
plt.subplot(1, 2, 2)
sns.barplot(x=churn_percentages.index, y=churn_percentages.values)
plt.title('Distribuição de Churn (Porcentagem)')
plt.xlabel('Churn')
plt.ylabel('Porcentagem de Clientes (%)')
for i, v in enumerate(churn_percentages.values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.close()

# Imprimir resultados
print("\nDistribuição de Churn:")
print("-" * 50)
print("\nNúmeros Absolutos:")
for status, count in churn_distribution.items():
    print(f"{'Clientes que saíram' if status else 'Clientes que permaneceram'}: {count:,}")

print("\nPorcentagens:")
for status, percentage in churn_percentages.items():
    print(f"{'Clientes que saíram' if status else 'Clientes que permaneceram'}: {percentage:.1f}%")

# Calcular razão de evasão
churn_ratio = churn_distribution[True] / churn_distribution[False]
print(f"\nRazão de Evasão (Saíram/Permaneceram): {churn_ratio:.2f}")
print(f"Para cada {int(1/churn_ratio)} clientes que permaneceram, {1:.0f} cliente saiu") 