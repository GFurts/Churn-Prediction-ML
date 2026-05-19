# Churn Prediction ML

Projeto de Machine Learning para previsão de churn de clientes em uma operadora de telecomunicações, desenvolvido com foco em portfólio e boas práticas de engenharia de ML.

## Objetivo

O objetivo deste projeto é prever quais clientes têm maior probabilidade de cancelar o serviço (**churn**) a partir de variáveis cadastrais, contratuais e de uso.

Além da modelagem preditiva, o projeto também inclui:
- comparação entre baselines e rede neural;
- rastreamento de experimentos com MLflow;
- API de inferência com FastAPI;
- testes automatizados;
- organização modular do código.

## Problema de negócio

Em empresas de telecomunicações, churn representa perda de receita e aumento do custo de aquisição de novos clientes. Antecipar clientes com maior risco de evasão permite criar ações de retenção mais direcionadas, reduzindo perdas e melhorando a eficiência operacional.

## Dataset

Foi utilizado o dataset **Telco Customer Churn**, com informações de clientes e sua condição final de permanência ou cancelamento.

Principais atributos utilizados:
- dados cadastrais;
- tipo de contrato;
- tempo de permanência (`tenure`);
- suporte técnico;
- serviços contratados;
- cobrança mensal e total.

## Etapas do projeto

### Etapa 1 — EDA e baselines
- análise exploratória dos dados;
- tratamento de qualidade dos dados;
- definição de métricas adequadas para classe desbalanceada;
- treinamento de baselines com:
  - DummyClassifier
  - Logistic Regression
- rastreamento com MLflow.

### Etapa 2 — Rede neural com PyTorch
- preparação dos dados para PyTorch;
- construção de uma MLP;
- treino com batching e early stopping;
- avaliação com:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
- comparação com os baselines.

### Etapa 3 — Engenharia e API
- refatoração do projeto para estrutura modular em `src/`;
- serialização do pipeline com `joblib`;
- criação de API com FastAPI;
- endpoints:
  - `/health`
  - `/predict`
- validação com Pydantic;
- testes com `pytest`;
- lint com `ruff`.

## Principais resultados

### Baselines
A Regressão Logística apresentou desempenho muito superior ao baseline ingênuo, mostrando que havia sinal preditivo relevante nos dados.

### MLP vs Logistic Regression
A MLP apresentou desempenho competitivo, com **recall ligeiramente superior**, enquanto a Regressão Logística manteve **precision mais alta**. As métricas de F1-score, ROC-AUC e PR-AUC ficaram bastante próximas entre os dois modelos.

### Decisão para serving
Para a API, foi escolhida a **Regressão Logística**, por apresentar desempenho competitivo e maior simplicidade operacional para uma primeira versão produtizável.

## Estrutura do projeto

```text
churn-prediction-ml/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md