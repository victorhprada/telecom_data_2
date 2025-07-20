import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Adicionar o diretório raiz ao path para importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR

# Carregar o modelo mais recente de Regressão Logística
lr_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('LogisticRegression_')]
latest_lr = max(lr_files, key=lambda x: x.split('_')[1])
model = joblib.load(os.path.join(MODELS_DIR, latest_lr))

# Carregar features
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train_transformed.csv'))
feature_names = X_train.columns.tolist()

# Obter coeficientes e criar DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': abs(model.coef_[0])
})

# Ordenar por magnitude absoluta
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

# Preparar o texto do relatório
report_lines = []
report_lines.append("Análise dos Coeficientes da Regressão Logística para Previsão de Churn")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Data da Análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"Modelo Utilizado: {latest_lr}")
report_lines.append("")
report_lines.append("Interpretação dos Coeficientes:")
report_lines.append("- Coeficientes positivos indicam maior probabilidade de churn")
report_lines.append("- Coeficientes negativos indicam menor probabilidade de churn")
report_lines.append("- A magnitude (valor absoluto) indica a força da influência")
report_lines.append("")
report_lines.append("Top 10 Features Mais Influentes:")
report_lines.append("-" * 50)

for _, row in coef_df.head(10).iterrows():
    direction = "aumenta" if row['Coefficient'] > 0 else "diminui"
    report_lines.append(f"\n{row['Feature']:<30} {row['Coefficient']:>10.4f}")
    report_lines.append(f"    → {direction} a probabilidade de churn")
    report_lines.append(f"    → Magnitude do efeito: {row['Abs_Coefficient']:.4f}")

report_lines.append("\nAgrupamento por Impacto:")
report_lines.append("-" * 50)

# Agrupar features por tipo de impacto
strong_positive = coef_df[coef_df['Coefficient'] > 0.5]['Feature'].tolist()
moderate_positive = coef_df[(coef_df['Coefficient'] <= 0.5) & (coef_df['Coefficient'] > 0.2)]['Feature'].tolist()
weak_positive = coef_df[(coef_df['Coefficient'] <= 0.2) & (coef_df['Coefficient'] > 0)]['Feature'].tolist()
strong_negative = coef_df[coef_df['Coefficient'] < -0.5]['Feature'].tolist()
moderate_negative = coef_df[(coef_df['Coefficient'] >= -0.5) & (coef_df['Coefficient'] < -0.2)]['Feature'].tolist()
weak_negative = coef_df[(coef_df['Coefficient'] >= -0.2) & (coef_df['Coefficient'] < 0)]['Feature'].tolist()

report_lines.append("\nForte Impacto Positivo (> 0.5):")
report_lines.extend([f"- {feature}" for feature in strong_positive])

report_lines.append("\nImpacto Positivo Moderado (0.2 - 0.5):")
report_lines.extend([f"- {feature}" for feature in moderate_positive])

report_lines.append("\nImpacto Positivo Fraco (0 - 0.2):")
report_lines.extend([f"- {feature}" for feature in weak_positive])

report_lines.append("\nForte Impacto Negativo (< -0.5):")
report_lines.extend([f"- {feature}" for feature in strong_negative])

report_lines.append("\nImpacto Negativo Moderado (-0.5 - -0.2):")
report_lines.extend([f"- {feature}" for feature in moderate_negative])

report_lines.append("\nImpacto Negativo Fraco (-0.2 - 0):")
report_lines.extend([f"- {feature}" for feature in weak_negative])

report_lines.append("\nRecomendações Baseadas nos Coeficientes:")
report_lines.append("-" * 50)

# Adicionar recomendações baseadas nos coeficientes mais significativos
top_positive = coef_df[coef_df['Coefficient'] > 0].head(3)['Feature'].tolist()
top_negative = coef_df[coef_df['Coefficient'] < 0].head(3)['Feature'].tolist()

report_lines.append("\nPara Reduzir Churn:")
report_lines.extend([f"- Fortalecer/Incentivar: {feature}" for feature in top_negative])

report_lines.append("\nMonitorar com Atenção:")
report_lines.extend([f"- Potencial Risco: {feature}" for feature in top_positive])

# Salvar o relatório
output_file = os.path.join(REPORTS_DIR, 'logistic_regression_analysis.txt')
with open(output_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"Análise salva em: {output_file}")
print("\nResumo dos Coeficientes Mais Importantes:")
print(coef_df[['Feature', 'Coefficient']].head(10).to_string(index=False)) 