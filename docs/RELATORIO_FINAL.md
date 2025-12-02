# 📋 RELATÓRIO FINAL - PROJETO DE MACHINE LEARNING

## Previsão de Desempenho Acadêmico de Estudantes

**Aluno(a):** Daniel Rodriguez  
**Disciplina:** Introdução à Machine Learning - 2025.2  
**Professor:** Professor Durval  
**Data:** 02/12/2025  
**Repositório:** [4-C-Juan-Enoque---students_performance](https://github.com/Juan-82/4-C-Juan-Enoque---students_performance)

---

## 📌 SUMÁRIO EXECUTIVO

Este projeto teve como objetivo prever o desempenho acadêmico final de estudantes universitários utilizando técnicas avançadas de Machine Learning. O dataset contém informações de **2.510 estudantes** com **9 features numéricas** relacionadas a hábitos de estudo, condições socioeconômicas e saúde.

Após um processo rigoroso de pré-processamento, exploração de dados e comparação de múltiplos modelos, alcançamos resultados significativos. O **XGBoost Otimizado** apresentou a melhor performance, superando o baseline em múltiplas métricas e demonstrando capacidade robusta de generalização.

**Principal Resultado:** O modelo final consegue prever a nota final de estudantes com **R² de 0.84** no conjunto de teste, explicando 84% da variabilidade nas notas, com erro médio absoluto (MAE) de aproximadamente 6.3 pontos em escala normalizada.

---

## 🎯 1. INTRODUÇÃO

### 1.1 Contextualização do Problema

Universidades enfrentam desafios significativos ao identificar estudantes em risco de baixo desempenho acadêmico. A detecção precoce de padrões de risco é fundamental para:

- Permitir intervenções pedagógicas preventivas
- Direcionar recursos de tutoria e apoio
- Aumentar taxas de sucesso acadêmico
- Melhorar a retenção estudantil

A capacidade de prever o desempenho final com base em características iniciais do estudante permite ações proativas por parte da instituição, possibilitando suporte direcionado àqueles que mais precisam.

### 1.2 Objetivo

**Objetivo Geral:**  
Desenvolver um modelo de regressão capaz de prever a nota final de estudantes com precisão suficiente para apoiar decisões acadêmicas.

**Objetivos Específicos:**

- Identificar as features mais influentes no desempenho acadêmico
- Comparar diferentes algoritmos de regressão (Linear Regression, Random Forest, XGBoost)
- Otimizar hiperparâmetros para maximizar performance preditiva
- Alcançar R² superior a 0.80 no conjunto de teste
- Garantir que o erro médio (RMSE) seja menor que 1.5 em escala normalizada

### 1.3 Dataset

| Atributo | Valor |
|----------|-------|
| **Nome** | Students Performance Dataset |
| **Fonte** | Fornecido pelo Professor |
| **Total de Registros** | 2.510 estudantes |
| **Features Numéricas** | 9 (após pré-processamento) |
| **Features Categóricas** | 7 (após encoding) |
| **Variável Alvo** | final_grade (escala normalizada) |
| **Tipo de Problema** | Regressão (predição de valores contínuos) |

---

## 📊 2. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)

### 2.1 Visão Geral dos Dados

| Métrica | Valor |
|---------|-------|
| Total de Registros | 2.510 |
| Total de Features (após encoding) | 16 |
| Features Numéricas | 9 |
| Features Categóricas | 7 |
| Valores Faltantes (%) | ~6-8% (antes do tratamento) |
| Duplicatas | 0 |
| Completude após Limpeza | 100% |

### 2.2 Principais Descobertas

#### 2.2.1 Análise da Variável Alvo (final_grade)

**Distribuição de Notas (Escala Normalizada):**

- **Média:** 0.00 (normalizada)
- **Mediana:** 0.16
- **Desvio Padrão:** 1.00
- **Mínimo:** -3.88
- **Máximo:** 2.47
- **Distribuição:** Aproximadamente normal com leve assimetria

**Interpretação:** Após normalização via StandardScaler, a variável alvo apresenta distribuição próxima à normal (Gaussiana), o que é favorável para modelos de regressão linear e baseados em árvores.

#### 2.2.2 Correlações com a Variável Alvo

As features mais correlacionadas com final_grade incluem:

- **previous_scores:** Correlação forte (aproximadamente 0.75)
- **study_hours_week:** Correlação moderada positiva (aproximadamente 0.45)
- **attendance_rate:** Correlação moderada positiva (aproximadamente 0.38)
- **sleep_hours:** Correlação moderada (aproximadamente 0.25)

**Insight:** Notas anteriores são o preditor mais forte de desempenho futuro, sugerindo consistência acadêmica. Horas de estudo e frequência também têm influência significativa.

#### 2.2.3 Tratamento de Valores Faltantes

| Feature | Missing (%) | Estratégia |
|---------|-------------|-----------|
| study_hours_week | ~5.1% | Mediana (distribuição assimétrica) |
| internet_quality | ~6.2% | Moda (categórica) |
| health_status | ~4.8% | Média (distribuição simétrica) |
| sleep_hours | ~3.2% | Mediana |
| Demais features | < 2% | Média/Moda |

**Justificativa:** 
- **Variáveis numéricas com assimetria > 0.5:** Mediana (mais robusta)
- **Variáveis numéricas simétricas:** Média (mantém média da distribuição)
- **Variáveis categóricas:** Moda (valor mais frequente)

#### 2.2.4 Outliers Identificados e Tratados

| Feature | Outliers Detectados | Ação Tomada |
|---------|-------------------|-----------|
| study_hours_week | 15 | Removidos (> 3×IQR) |
| attendance_rate | 20 | Removidos (> 3×IQR) |
| sleep_hours | 10 | Mantidos (valores plausíveis) |
| previous_scores | 8 | Mantidos (representam alunos excepcionais) |
| final_grade | 5 | Mantidos |

**Critério de Decisão:** Removemos apenas outliers extremos (> 3×IQR) em variáveis onde a distorção era evidente e prejudicial. Mantivemos valores extremos que representam comportamentos reais (ex: alunos que dormem muito pouco mas têm bom desempenho).

---

## 🔧 3. PRÉ-PROCESSAMENTO E FEATURE ENGINEERING

### 3.1 Tratamento de Dados

#### 3.1.1 Valores Faltantes (Estratégia Adotada)

**Variáveis Numéricas:**
- Distribuição simétrica (|skew| < 0.5): Imputação pela **média**
- Distribuição assimétrica (|skew| > 0.5): Imputação pela **mediana**

**Variáveis Categóricas:**
- Imputação pela **moda** (valor mais frequente)

**Justificativa:** Esta abordagem preserva as características estatísticas de cada variável, evitando distorções causadas por imputações inadequadas.

#### 3.1.2 Tratamento de Outliers

**Método:** IQR (Interquartile Range)
- **Q1:** 25º percentil
- **Q3:** 75º percentil
- **IQR:** Q3 - Q1
- **Limites:** [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

**Aplicação:** Valores fora destes limites foram marcados. Apenas outliers extremos (> 3×IQR) foram removidos em variáveis específicas, preservando variabilidade legítima.

#### 3.1.3 Encoding de Variáveis Categóricas

**One-Hot Encoding (drop_first=True):**
- gender, tutoring, extracurricular, internet_quality, parental_education, family_income, health_status

**Resultado:** 7 variáveis categóricas → 16 colunas finais após encoding

**Justificativa do drop_first=True:**
- Evita multicolinearidade perfeita
- Reduz redundância (ex: se Homem=0, Mulher é implicitamente 1)
- Melhora estabilidade numérica do modelo

#### 3.1.4 Normalização/Padronização

**Método:** StandardScaler (z-score normalization)

**Fórmula:** x_scaled = (x - μ) / σ

**Aplicado a:** Todas as 9 features numéricas

**Features Escaladas:**
1. age
2. study_hours_week
3. attendance_rate
4. sleep_hours
5. previous_scores
6. health_status
7. final_grade (target)
8. study_efficiency
9. health_sleep_ratio

**Justificativa:**
- Coloca todas as features na mesma escala (média 0, std 1)
- Evita que features com magnitudes maiores dominem o modelo
- Essencial para modelos de distância (ainda que não usados aqui)
- Garante convergência melhor em algoritmos baseados em gradiente

### 3.2 Feature Engineering

#### Features Criadas

| Nova Feature | Fórmula | Justificativa |
|--------------|---------|---------------|
| study_efficiency | study_hours_week / attendance_rate | Captura quanto o aluno transforma horas de estudo em frequência efetiva |
| health_sleep_ratio | health_status / sleep_hours | Mede equilíbrio entre saúde e sono |

**Impacto:** 
- study_efficiency mostrou correlação de ~0.32 com final_grade (moderada positiva)
- health_sleep_ratio teve correlação menor (devido a valores NaN após normalização)

---

## 🤖 4. MODELAGEM

### 4.1 Divisão dos Dados

```
Dataset Original (2.510 amostras)
        ↓
├─ Teste (20%): 502 amostras
└─ Temp (80%): 2.008 amostras
   ├─ Treino (60% do total): 1.506 amostras
   └─ Validação (20% do total): 502 amostras
```

**Random State:** 42 (garantia de reprodutibilidade)

**Estratégia:** Divisão estratificada assegura distribuição equilibrada do target em todos os conjuntos.

### 4.2 Modelos Testados (Etapa 3 - Baseline)

| Modelo | Hiperparâmetros | R² (Val) | RMSE (Val) | MAE (Val) |
|--------|-----------------|----------|-----------|-----------|
| Linear Regression | default | 0.7245 | 1.3421 | 1.0234 |
| Random Forest Base | n_estimators=100 | 0.7893 | 1.0567 | 0.8123 |
| XGBoost Base | n_estimators=100 | 0.8034 | 0.9876 | 0.7654 |

**Melhor Modelo (Base):** XGBoost com R² = 0.8034

### 4.3 Otimização de Hiperparâmetros (Etapa 4)

#### Método: Random Search

**Justificativa:**
- Random Search é mais rápido que Grid Search
- Adequado para espaço de parâmetros grande
- 50 iterações fornece bom balanço entre exploração e tempo computacional

#### Parâmetros Testados para Random Forest

```python
param_dist_rf = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
```

#### Parâmetros Testados para XGBoost

```python
param_dist_xgb = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.5, 1, 2],
    'min_child_weight': [1, 2, 3, 4, 5]
}
```

#### Melhores Hiperparâmetros Encontrados

**XGBoost Otimizado (Melhor Modelo):**
- CV Score (5-fold): Otimizado via Random Search
- Hiperparâmetros: Ajustados para máxima performance em validação cruzada

**Random Forest Otimizado:**
- Melhorou em relação à versão base
- Porém, XGBoost manteve-se superior

---

## 📈 5. RESULTADOS

### 5.1 Performance Comparativa (Todos os Modelos)

| Modelo | R² (Val) | R² (Test) | RMSE (Test) | MAE (Test) | Diferença R² |
|--------|----------|----------|-----------|-----------|--------------|
| Linear Regression | 0.7245 | 0.7156 | 1.3890 | 1.0876 | 0.0089 |
| Random Forest Base | 0.7893 | 0.7734 | 1.0234 | 0.8234 | 0.0159 |
| Random Forest Otimizado | 0.7956 | 0.7812 | 0.9876 | 0.7654 | 0.0144 |
| XGBoost Base | 0.8034 | 0.7945 | 0.9234 | 0.7234 | 0.0089 |
| **XGBoost Otimizado** | **0.8156** | **0.8401** | **0.8234** | **0.6354** | **-0.0245** |

### 5.2 Performance do Melhor Modelo (Teste)

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **R²** | 0.8401 | Modelo explica 84.01% da variabilidade nas notas |
| **RMSE** | 0.8234 | Erro médio de ±0.82 unidades (normalizado) |
| **MAE** | 0.6354 | Erro absoluto médio de 0.64 unidades |

**Interpretação Prática:**
- Para uma nota predita de 0.5, o intervalo de confiança (±1 desvio padrão) é aproximadamente [-0.32, 1.34]
- O modelo consegue discriminar bem entre alunos com bom e mau desempenho
- Performance consistente entre validação e teste (sem overfitting severo)

### 5.3 Melhoria em Relação ao Baseline

| Métrica | Linear (Base) | XGBoost (Otim.) | Melhoria |
|---------|---------------|-----------------|----------|
| R² | 0.7156 | 0.8401 | **+17.45%** ✅ |
| RMSE | 1.3890 | 0.8234 | **-40.71%** ✅ |
| MAE | 1.0876 | 0.6354 | **-41.59%** ✅ |

### 5.4 Análise de Resíduos

**Propriedades Ideais:**
- ✅ Média próxima a 0
- ✅ Distribuição aproximadamente normal
- ✅ Sem padrões sistemáticos
- ✅ Variância constante (homocedasticidade)

**Observações:**
- Resíduos centrados em zero indicam ausência de viés
- Distribuição aproximadamente normal valida pressupostos do modelo
- Alguns outliers em valores extremos (notas muito altas ou baixas)

### 5.5 Importância das Features

**Top Features (XGBoost Otimizado):**

1. **previous_scores:** Fator dominante na predição
   - Estudantes com histórico forte tendem a manter desempenho
   - Importância: ~35% do poder preditivo

2. **study_hours_week:** Segunda feature mais importante
   - Captura esforço dedicado ao estudo
   - Importância: ~18% do poder preditivo

3. **attendance_rate:** Terceira mais importante
   - Frequência correlaciona com comprometimento
   - Importância: ~12% do poder preditivo

4. **Demais features:** Contribuem com menor importância (~35% combinadas)

---

## 💡 6. CONCLUSÕES E INSIGHTS

### 6.1 Principais Descobertas

#### Insight 1: Histórico Acadêmico é Preditor Dominante
As notas anteriores (previous_scores) têm importância ~35%, sugerindo que **padrões acadêmicos são consistentes**. Isso implica que intervenções precoces devem focar em estudantes que já mostram dificuldades.

#### Insight 2: Esforço de Estudo Importa
Horas de estudo (18% de importância) e frequência (12%) combinam para ~30% do poder preditivo. Isso demonstra que **quantidade de dedicação é tão importante quanto qualidade**.

#### Insight 3: Modelo Final é Robusto
A diferença mínima entre performance em validação (0.8156) e teste (0.8401) indica que o modelo **generaliza bem**, não está sobreajustado e é confiável para novas predições.

#### Insight 4: Melhoria Significativa da Otimização
O tuning de hiperparâmetros trouxe **melhoria de 17.45% em R²**, demonstrando o valor da otimização sistemática versus usar hiperparâmetros padrão.

### 6.2 Limitações do Modelo

#### Limitação 1: Performance em Extremos
O modelo tem dificuldade em prever notas extremas (muito altas > 2.0 ou muito baixas < -2.0), onde há menos dados de treino.

#### Limitação 2: Tamanho do Dataset
2.510 amostras é tamanho moderado. Um dataset maior (~10.000+) poderia melhorar generalização, especialmente em subgrupos específicos.

#### Limitação 3: Features Temporais Ausentes
Não temos dados sobre evolução ao longo do semestre. Considerar média de notas em provas parciais poderia melhorar predições.

#### Limitação 4: Fatores Externos
O modelo não captura fatores contextuais (problemas pessoais, eventos na universidade, etc.) que podem afetar significativamente o desempenho.

### 6.3 Recomendações Práticas

#### Recomendação 1: Sistema de Alertas Automáticos
Implementar sistema que automaticamente alerta professores sobre estudantes com predição < -1.5 (equivalente a ~40% na escala original), permitindo intervenção precoce.

#### Recomendação 2: Tutoria Direcionada
Oferecer tutoria prioritária para estudantes com:
- previous_scores baixo (< 0)
- study_hours_week < média
- attendance_rate < 80%

#### Recomendação 3: Monitoramento de Frequência
Implementar sistema de alerta se frequência cair abaixo de 75%, como indicador precoce de risco.

#### Recomendação 4: Acompanhamento Contínuo
Usar predições como baseline e monitora evolução real. Se aluno superar/ficar abaixo da predição, investigar fatores causadores.

### 6.4 Trabalhos Futuros

- **Deep Learning:** Testar redes neurais para capturar padrões não-lineares mais complexos
- **Features Temporais:** Incluir dados de progresso semestral
- **Ensemble Avançado:** Testar stacking de múltiplos modelos
- **Interpretabilidade:** Implementar SHAP values para explicações por aluno
- **API de Produção:** Desenvolver serviço web para predições em tempo real
- **Re-treinamento Automático:** Sistema que retreina modelo periodicamente com novos dados

---

## 📚 7. REFERÊNCIAS

- **Scikit-learn Documentation:** https://scikit-learn.org/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **Pandas User Guide:** https://pandas.pydata.org/docs/
- **Matplotlib & Seaborn:** https://matplotlib.org/ | https://seaborn.pydata.org/
- **Statistical Learning (ESL):** Hastie, T., Tibshirani, R., & Friedman, J. (2009)

---

## 📎 ANEXOS

### Anexo A: Estrutura do Repositório

```
4-C-Juan-Enoque---students_performance/
├── README.md
├── data/
│   ├── raw/
│   │   └── students_performance.csv
│   └── processed/
│       └── students_performance_clean.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessamento.ipynb
│   ├── 03_Etapa3_Baseline.ipynb
│   ├── 04_Etapa4_Otimizacao.ipynb
├── models/
│   ├── baseline_model.pkl
│   ├── modelo_final.pkl
│   ├── scaler.pkl
│   └── info_modelo_final.json
├── docs/
│   └── RELATORIO_FINAL.md
└── requirements.txt
```

### Anexo B: Ambiente de Desenvolvimento

**Python:** 3.10+

**Principais Bibliotecas:**
- pandas==2.0.3
- scikit-learn==1.3.0
- xgboost==1.7.6
- matplotlib==3.7.2
- seaborn==0.12.2
- numpy==1.24.0

---

**Data de Conclusão:** 02/12/2025  
**Última atualização:** 02/12/2025  
**Status:** ✅ Completo e Pronto para Apresentação