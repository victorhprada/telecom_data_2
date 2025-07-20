# Análise de Churn em Telecomunicações

## Sumário Executivo

Este projeto analisa e prediz a evasão de clientes (churn) em uma empresa de telecomunicações, utilizando técnicas de machine learning e análise estatística. O objetivo é identificar os principais fatores que influenciam a evasão e propor estratégias de retenção baseadas em evidências.

### Principais Resultados
- Taxa atual de churn: 28.8%
- Acurácia dos modelos: ~78%
- ROC AUC: ~82%

## Estrutura do Projeto

```
telecom_data_2/
│
├── data/               # Dados brutos e processados
├── models/            # Modelos treinados
├── notebooks/         # Jupyter notebooks
├── reports/           # Relatórios e visualizações
└── src/               # Código fonte
```

## Análise Detalhada do Churn

### 1. Fatores Mais Influentes (por ordem de importância)

#### Características Contratuais
1. **Tipo de Contrato** (Coef: -0.88 para contratos de 2 anos)
   - Contratos mensais: 44.6% de churn
   - Contratos anuais: 14.0% de churn
   - Contratos de 2 anos: 5.5% de churn
   - **Impacto**: Contratos mais longos reduzem significativamente o churn

2. **Tempo como Cliente** (Coef: -0.75)
   - Média para clientes que permanecem: 37.6 meses
   - Média para clientes que saem: 19.4 meses
   - **Impacto**: Maior tempo como cliente reduz probabilidade de churn

#### Características de Serviço
3. **Serviço de Internet** (Coef: 0.48 para fibra)
   - Fibra ótica tem maior taxa de churn
   - **Impacto**: Clientes com fibra têm 48% mais chance de churn

4. **Serviços de Segurança**
   - Segurança Online (Coef: -0.38)
   - Suporte Técnico (Coef: -0.40)
   - **Impacto**: Serviços de proteção reduzem churn

#### Características Financeiras
5. **Gastos Mensais** (Coef: 0.42)
   - Média para clientes que saem: $73.26
   - Média para clientes que ficam: $61.27
   - **Impacto**: Valores mais altos aumentam risco de churn

### 2. Desempenho dos Modelos

#### Regressão Logística
- Accuracy: 78.06%
- Precision: 64.53%
- Recall: 52.98%
- F1-Score: 58.19%
- ROC AUC: 81.72%

#### Random Forest
- Accuracy: 77.99%
- Precision: 64.86%
- Recall: 51.55%
- F1-Score: 57.45%
- ROC AUC: 81.84%

### 3. Estratégias de Retenção Recomendadas

#### Estratégia 1: Fidelização por Contrato
- **Objetivo**: Aumentar adesão a contratos mais longos
- **Ações**:
  - Oferecer descontos progressivos para contratos mais longos
  - Criar benefícios exclusivos para contratos anuais e bianuais
  - Desenvolver programa de pontos/recompensas vinculado ao tempo de contrato

#### Estratégia 2: Otimização de Preços
- **Objetivo**: Reduzir churn relacionado a custos
- **Ações**:
  - Revisar política de preços para clientes com faturas altas
  - Criar pacotes com melhor custo-benefício
  - Implementar alertas de uso para evitar faturas inesperadas

#### Estratégia 3: Melhoria de Serviços
- **Objetivo**: Aumentar satisfação com serviços de fibra
- **Ações**:
  - Fortalecer suporte técnico para clientes de fibra
  - Implementar monitoramento proativo de qualidade
  - Oferecer pacote de segurança online gratuito no primeiro ano

#### Estratégia 4: Programa de Retenção Preventiva
- **Objetivo**: Identificar e reter clientes em risco
- **Ações**:
  - Monitorar indicadores de risco (alto valor de fatura, problemas técnicos)
  - Criar equipe especializada em retenção preventiva
  - Desenvolver ofertas personalizadas baseadas no perfil de risco

### 4. KPIs para Monitoramento

1. **Métricas de Churn**
   - Taxa mensal de churn
   - Tempo médio de permanência
   - Taxa de conversão para contratos longos

2. **Métricas de Serviço**
   - Satisfação com internet fibra
   - Taxa de adoção de serviços de segurança
   - Tempo de resolução de problemas técnicos

3. **Métricas Financeiras**
   - Valor médio da fatura
   - Taxa de inadimplência
   - ROI das ações de retenção

## Conclusão

A análise revelou que o churn é fortemente influenciado por fatores contratuais e financeiros. As estratégias propostas focam em aumentar o compromisso de longo prazo dos clientes enquanto otimizam a relação custo-benefício dos serviços.

## Próximos Passos

1. Implementar sistema de scoring de risco de churn
2. Desenvolver dashboard de monitoramento em tempo real
3. Realizar testes A/B das estratégias de retenção
4. Refinar modelos com feedback das ações implementadas

## Uso do Projeto

```bash
# Processamento de dados
python src/preprocessing/data_processor.py

# Análise de features
python src/analysis/feature_analysis.py

# Treinamento de modelos
python src/models/model_trainer.py
```

## Dependências

Ver `requirements.txt` para lista completa de dependências.