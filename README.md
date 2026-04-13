# Análise de Crédito em Dados do LendingClub

### EDA · Classificação · Random Forest · ROC-AUC · FastAPI · Docker · Deploy

&nbsp;

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/API-online-28a745?style=for-the-badge)](https://api-credito-lendingclub.onrender.com)

&nbsp;
> Pipeline completo de Machine Learning para previsão de inadimplência em empréstimos pessoais,
> com foco em tratamento de dados desbalanceados, métricas adequadas para risco de crédito
> e deploy em produção com API REST containerizada.

&nbsp;

### Interface Interativa

[![Interface do Modelo](assets/modelo_em_funcionamento.png)](https://api-credito-lendingclub.onrender.com/app)

> Acesse a interface em: **[api-credito-lendingclub.onrender.com/app](https://api-credito-lendingclub.onrender.com/app)**

### Documentação Swagger

[![Swagger UI](assets/Swagger_UI.png)](https://api-credito-lendingclub.onrender.com/docs)

> Documentação completa da API em: **[api-credito-lendingclub.onrender.com/docs](https://api-credito-lendingclub.onrender.com/docs)**
---

## Índice

- [Contexto](#contexto)
- [Objetivos](#objetivos)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Análise Exploratória](#análise-exploratória)
- [Modelos Avaliados](#modelos-avaliados)
- [Principais Resultados](#principais-resultados)
- [API em Produção](#api-em-produção)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Autor](#autor)

---

## Contexto

Projeto de Machine Learning aplicado ao mercado de crédito, utilizando dados históricos da plataforma LendingClub que conecta tomadores e investidores de empréstimos pessoais. O modelo identifica mutuários com maior risco de não pagamento, apoiando decisões de concessão de crédito. O pipeline completo vai da análise exploratória ao deploy em produção como API REST containerizada.

| Etapa | Descrição |
|---|---|
| **EDA** | Análise do perfil FICO, taxas de juros e finalidades de empréstimo |
| **Encoding** | One-Hot Encoding na variável categórica `finalidade` |
| **Modelagem** | Decision Tree como baseline e Random Forest como modelo final |
| **Balanceamento** | `class_weight='balanced'` para tratar desbalanceamento de classes |
| **Avaliação** | ROC-AUC, curva ROC e importância de features |
| **Deploy** | API REST com FastAPI + Docker + Render |

---

## Objetivos

- Construir um modelo de classificação para prever inadimplência em empréstimos
- Comparar Decision Tree e Random Forest em dataset desbalanceado (~85% adimplentes)
- Aplicar `class_weight='balanced'` para melhorar a detecção de inadimplentes
- Avaliar com ROC-AUC métrica mais adequada que acurácia em problemas de crédito
- Criar uma API REST com FastAPI e containerizar com Docker
- Fazer deploy em produção com link público acessível

---

## Pipeline do Projeto

```mermaid
flowchart TD
    A([Dataset\nLendingClub\n2007–2010]) --> B[EDA\nFICO · Taxa de Juros · Finalidade]
    B --> C[Encoding\nOne-Hot em Finalidade]
    C --> D[Split Treino/Teste\n70% / 30%]
    D --> E[Decision Tree\nBaseline]
    D --> F[Random Forest\nclass_weight=balanced]
    E --> G[Avaliação\nROC-AUC · Classification Report]
    F --> G
    G --> H[API REST\nFastAPI · Docker]
    H --> I([Deploy\nRender · Link público])

    style A fill:#4A90D9,color:#fff,stroke:none
    style I fill:#28a745,color:#fff,stroke:none
    style B fill:#6C757D,color:#fff,stroke:none
    style C fill:#6C757D,color:#fff,stroke:none
    style D fill:#6C757D,color:#fff,stroke:none
    style E fill:#6C757D,color:#fff,stroke:none
    style F fill:#6C757D,color:#fff,stroke:none
    style G fill:#6C757D,color:#fff,stroke:none
    style H fill:#6C757D,color:#fff,stroke:none
```

---

## Tecnologias Utilizadas

| Tecnologia | Uso no Projeto |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Linguagem principal |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Manipulação e análise dos dados |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Operações numéricas |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | Modelos, métricas e curva ROC |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white) | Visualizações e curva ROC |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white) | Histogramas e gráficos exploratórios |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | API REST para servir o modelo em produção |
| ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | Containerização da aplicação |
| ![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white) | Hospedagem do deploy em produção |

---

## Dataset

**Fonte:** [LendingClub](https://www.lendingclub.com) dados de empréstimos 2007–2010
**Uso:** Exclusivamente educacional

| Característica | Detalhe |
|---|---|
| Volume | ~9.578 empréstimos |
| Período | 2007 a 2010 |
| Variável target | `nao_pago_totalmente` (1 = inadimplente) |
| Desbalanceamento | ~85% adimplentes / ~15% inadimplentes |

**Principais variáveis:**

| Variável | Descrição |
|---|---|
| `pontuacao_fico` | Score de crédito do mutuário |
| `taxa_juros` | Taxa anual do empréstimo |
| `finalidade` | Objetivo do empréstimo (cartão, educação, negócio...) |
| `relacao_divida_renda` | Razão dívida/renda |
| `log_renda_anual` | Log da renda anual declarada |
| `politica_credito` | Se atende aos critérios da plataforma |

---

## Análise Exploratória

### Inadimplência por Finalidade do Empréstimo

![Countplot Finalidade](assets/countplot_finalidade_inadimplencia.png)

> Empréstimos para pequenos negócios concentram maior proporção de inadimplentes em relação ao total de solicitações indicando maior risco por finalidade. Consolidação de dívidas é a categoria com maior volume absoluto.

### Distribuição FICO por Adimplência

![Histograma FICO](assets/histograma_fico_adimplencia.png)

> Inadimplentes (vermelho) apresentam distribuição FICO deslocada para scores mais baixos em relação aos adimplentes (azul) confirmando que a pontuação de crédito é um dos principais preditores de risco.

### FICO vs Taxa de Juros por Política de Crédito

![Lmplot FICO Juros](assets/lmplot_fico_juros_politica.png)

> Relação inversa clara entre pontuação FICO e taxa de juros clientes com maior score recebem taxas menores. Clientes que não atendem à política de crédito (vermelho) concentram scores baixos e juros altos, independentemente do status de pagamento.

### Curva ROC Comparação de Modelos

![Curva ROC](assets/curva_roc.png)

| Modelo | ROC-AUC | Recall Classe 1 (inadimplente) |
|---|---|---|
| Decision Tree | menor | ~22% |
| **Random Forest (balanced)** | **maior** | **melhorado** |

> Em problemas de crédito, **liberar crédito para inadimplentes gera prejuízo direto**. Por isso o Recall da classe 1 e o ROC-AUC são as métricas prioritárias não a acurácia geral.

### Features Mais Importantes

![Feature Importance](assets/feature_importance_credito.png)

> A pontuação FICO e a taxa de juros lideram em importância confirmando que o histórico de crédito e o risco percebido pelo mercado são os sinais mais fortes de inadimplência. A parcela mensal e a renda anual também contribuem significativamente.

---

## Principais Resultados

### Por que Random Forest com class_weight='balanced'?

| Aspecto | Decision Tree | Random Forest (balanced) |
|---|---|---|
| Acurácia geral | ~85% | ~85% |
| Recall inadimplentes | ~22% | melhorado |
| ROC-AUC | menor | **maior** |
| Estabilidade | baixa | alta |

> O `class_weight='balanced'` penaliza mais os erros na classe minoritária (inadimplentes), forçando o modelo a identificá-los melhor comportamento essencial em problemas de crédito onde **liberar crédito para inadimplentes gera prejuízo direto**.

### Aplicações do Modelo

- Apoio à decisão de concessão de crédito em fintechs e bancos
- Score de risco para triagem automática de solicitações
- Base para definição de limites de crédito por perfil de risco
- Deploy via API para integração com sistemas de análise de crédito

---

## API em Produção

### Interface Interativa

> Acesse a interface em: **[api-credito-lendingclub.onrender.com/app](https://api-credito-lendingclub.onrender.com/app)**

### Exemplo de Requisição

```bash
curl -X POST https://api-credito-lendingclub.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "politica_credito": 1,
    "taxa_juros": 0.12,
    "parcela_mensal": 350.0,
    "log_renda_anual": 10.5,
    "relacao_divida_renda": 15.0,
    "pontuacao_fico": 700,
    "dias_com_linha_credito": 3000,
    "saldo_rotativo": 10000,
    "utilizacao_rotativa": 50.0,
    "consultas_ultimos_6meses": 0,
    "atrasos_ultimos_2anos": 0,
    "registros_publicos": 0,
    "finalidade": "debt_consolidation"
  }'
```

### Resposta

```json
{
  "inadimplente": 0,
  "resultado": "Baixo risco de inadimplência",
  "probabilidade_inadimplencia": 0.1823,
  "probabilidade_pagamento": 0.8177,
  "modelo": "RandomForestClassifier"
}
```

### Endpoints disponíveis

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Status da API |
| `GET` | `/app` | Interface interativa |
| `GET` | `/docs` | Documentação Swagger |
| `POST` | `/predict` | Análise de risco de crédito |

---

## Estrutura do Repositório

```
Analise_de_credito_em_dados_do_LendingClub/
│
├──  assets/                                   # Gráficos gerados na análise
│   ├── countplot_finalidade_inadimplencia.png
│   ├── histograma_fico_adimplencia.png
│   ├── lmplot_fico_juros_politica.png
│   ├── curva_roc.png
│   └── feature_importance_credito.png
│   └── modelo_em_funcionamento.png
│   └── Swagger_UI.png
├──  analise_de_credito_LendingClub.ipynb      # Notebook completo
├──  main.py                                   # API FastAPI
├──  index.html                                # Interface interativa
├──  Dockerfile                                # Containerização
├──  modelo_credito_lendingclub.pkl            # Modelo Random Forest treinado
├──  colunas_credito.pkl                       # Features esperadas pela API
├──  loan_data.csv                             # Dataset original
├──  requirements.txt                          # Dependências do projeto
└──  README.md                                 # Documentação do projeto
```

---

## Autor

<div align="center">

<img src="https://github.com/Anderson1999DC.png" width="100px" style="border-radius:50%"/>

**Anderson Coelho**
*Cientista de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anderson-coelho-42671634a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anderson1999DC)

</div>

---

<div align="center">
