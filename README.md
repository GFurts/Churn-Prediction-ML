# Churn Prediction ML

Projeto de **Machine Learning para previsão de churn** de clientes em uma operadora de telecomunicações, desenvolvido com foco em **portfólio** e **boas práticas de engenharia de ML**.

---

## Overview

Este projeto tem como objetivo prever quais clientes têm maior probabilidade de cancelar o serviço (**churn**) a partir de variáveis cadastrais, contratuais e de uso.

Além da modelagem preditiva, o projeto também inclui:

- Análise exploratória dos dados;
- Comparação entre baselines e rede neural;
- Rastreamento de experimentos com **MLflow**;
- API de inferência com **FastAPI**;
- Testes automatizados com **pytest**;
- Organização modular do código em `src/`.

---

## Business Problem

Em empresas de telecomunicações, churn representa:

- Perda de receita;
- Aumento do custo de aquisição de novos clientes;
- Necessidade de campanhas de retenção mais eficientes.

Antecipar clientes com maior risco de evasão permite criar ações mais direcionadas e reduzir perdas operacionais.

---

## Dataset

Foi utilizado o dataset **Telco Customer Churn**, contendo informações de clientes e sua condição final de permanência ou cancelamento.

### Principais atributos utilizados

- Dados cadastrais;
- Tipo de contrato;
- Tempo de permanência (`tenure`);
- Suporte técnico;
- Serviços contratados;
- Cobrança mensal (`MonthlyCharges`);
- Cobrança total (`TotalCharges`).

---

## Project Stages

### Stage 1 — EDA and Baselines

Nesta etapa foram realizados:

- Análise exploratória dos dados;
- Tratamento de qualidade dos dados;
- Definição de métricas adequadas para classe desbalanceada;
- Treinamento de modelos baseline:
  - `DummyClassifier`
  - `LogisticRegression`
- Rastreamento dos experimentos com **MLflow**.

---

### Stage 2 — Neural Network with PyTorch

Nesta etapa foi construída uma **MLP em PyTorch**, com:

- Preparação dos dados para entrada na rede;
- Construção da arquitetura da MLP;
- Treino com **batching** e **early stopping**;
- Avaliação com:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
- Comparação com os modelos baseline.

---

### Stage 3 — ML Engineering and API

Nesta etapa o projeto foi refatorado para uma estrutura mais próxima de produção:

- Modularização do código em `src/`;
- Serialização do pipeline com `joblib`;
- Criação de API com **FastAPI**;
- Endpoints:
  - `/health`
  - `/predict`
- Validação de payload com **Pydantic**;
- Testes automatizados com **pytest**;
- Lint com **ruff**.

---

## Main Results

### Baselines

A **Regressão Logística** apresentou desempenho muito superior ao baseline ingênuo, mostrando que havia sinal preditivo relevante nos dados.

### MLP vs Logistic Regression

A **MLP** apresentou desempenho competitivo, com **recall ligeiramente superior**, enquanto a **Regressão Logística** manteve **precision mais alta**.

As métricas de:

- F1-score
- ROC-AUC
- PR-AUC

ficaram bastante próximas entre os dois modelos.

### Serving Decision

Para a API, foi escolhida a **Regressão Logística**, por apresentar:

- Desempenho competitivo;
- Maior simplicidade operacional;
- Maior facilidade de manutenção para uma primeira versão produtizável.

A MLP foi mantida como modelo avançado de comparação experimental.

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **PyTorch**
- **MLflow**
- **FastAPI**
- **Pytest**
- **Ruff**
- **Joblib**
- **Jupyter Notebook**

---

## Project Structure

```text
churn-prediction-ml/
├── data/
│   └── raw/
├── docs/
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
API Endpoints
GET /health
Retorna o status da API.

Exemplo de resposta:

JSON
{
  "status": "ok"
}
POST /predict
Recebe os dados de um cliente e retorna a classe prevista (prediction) e a probabilidade de churn (churn_probability).

Exemplo de payload:

JSON
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
Exemplo de resposta:

JSON
{
  "prediction": 1,
  "churn_probability": 0.7138
}
How to Run the Project
1. Clone the repository
Bash
git clone [https://github.com/GFurts/Churn-Prediction-ML.git](https://github.com/GFurts/Churn-Prediction-ML.git)
cd churn-prediction-ml
2. Create and activate a virtual environment
Bash
python -m venv .venv
Windows:

Bash
.venv\Scripts\activate
3. Install dependencies
Bash
pip install -r requirements.txt
4. Train the baseline model served by the API
Bash
python -m src.train_baseline
5. Run the API
Bash
uvicorn src.api:app --reload
6. Access interactive documentation
Abra no navegador: http://127.0.0.1:8000/docs

Running Tests
Bash
pytest -v
Code Quality
Bash
ruff check .
Why This Project Matters for My Portfolio
Este projeto foi construído para demonstrar competências em três frentes:

Análise de dados e modelagem preditiva clássica.

Deep Learning aplicado estruturado com PyTorch.

Engenharia de ML, englobando o desenvolvimento de API, testes automatizados, linters e uma arquitetura de código totalmente modular.

Ele representa não apenas a capacidade de construir modelos matemáticos isolados, mas também a habilidade de transformar um experimento de laboratório em um microsserviço estruturado próximo ao padrão de produção.

Next Steps
Possíveis evoluções futuras mapeadas para o projeto:

Adicionar monitoramento de predições e detecção de Data Drift;

Testar o serving alternativo da MLP em uma versão futura da API;

Melhorar a documentação de arquitetura de software;

Realizar o deploy em ambiente de nuvem;

Adicionar esteiras de CI/CD via GitHub Actions.

Author
Desenvolvido por Gabriel Furtado.