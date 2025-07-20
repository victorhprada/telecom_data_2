import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados preparados
print("Carregando os dados...")
X_train = pd.read_csv('data/X_train_transformed.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test_transformed.csv')
y_test = pd.read_csv('data/y_test.csv')

# Função para plotar distribuição
def plot_distribution(y, title, ax):
    sns.countplot(x=y, ax=ax)
    ax.set_title(title)
    total = len(y)
    for i, v in enumerate(Counter(y).values()):
        ax.text(i, v, f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom')

# Criar figura para visualização
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparação das Técnicas de Balanceamento', fontsize=16)

# Distribuição original
print("\nDistribuição original dos dados:")
print(Counter(y_train['Churn']))
plot_distribution(y_train['Churn'], 'Distribuição Original', axes[0,0])

# Aplicar SMOTE (Oversampling)
print("\nAplicando SMOTE (Oversampling)...")
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train['Churn'])
print("Distribuição após SMOTE:")
print(Counter(y_smote))
plot_distribution(y_smote, 'Após SMOTE', axes[0,1])

# Aplicar Random Undersampling
print("\nAplicando Random Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train['Churn'])
print("Distribuição após Undersampling:")
print(Counter(y_under))
plot_distribution(y_under, 'Após Undersampling', axes[1,0])

# Aplicar SMOTETomek (Combinação de over e undersampling)
print("\nAplicando SMOTETomek...")
smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X_train, y_train['Churn'])
print("Distribuição após SMOTETomek:")
print(Counter(y_smt))
plot_distribution(y_smt, 'Após SMOTETomek', axes[1,1])

plt.tight_layout()
plt.savefig('balancing_comparison.png')
plt.close()

# Salvar os diferentes conjuntos de dados balanceados
print("\nSalvando os dados balanceados...")

# SMOTE
pd.DataFrame(X_smote, columns=X_train.columns).to_csv('data/X_train_smote.csv', index=False)
pd.DataFrame(y_smote, columns=['Churn']).to_csv('data/y_train_smote.csv', index=False)

# Undersampling
pd.DataFrame(X_under, columns=X_train.columns).to_csv('data/X_train_undersampled.csv', index=False)
pd.DataFrame(y_under, columns=['Churn']).to_csv('data/y_train_undersampled.csv', index=False)

# SMOTETomek
pd.DataFrame(X_smt, columns=X_train.columns).to_csv('data/X_train_smotetomek.csv', index=False)
pd.DataFrame(y_smt, columns=['Churn']).to_csv('data/y_train_smotetomek.csv', index=False)

print("\nComparação das dimensões dos dados:")
print(f"Original: {X_train.shape}")
print(f"SMOTE: {X_smote.shape}")
print(f"Undersampling: {X_under.shape}")
print(f"SMOTETomek: {X_smt.shape}")

print("\nProcesso de balanceamento concluído! Os dados balanceados foram salvos na pasta 'data/'")
print("Um gráfico comparativo foi salvo como 'balancing_comparison.png'") 