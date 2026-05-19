import pandas as pd


def preparar_dados_para_treino(df):
    """
    Função completa de pré-processamento para o Tech Challenge.
    """
    # 1. Limpeza de tipos (Célula 5 e 6 do seu notebook)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # 2. Remover ID (Não serve para o modelo aprender)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # 3. Converter o Alvo (Churn) para números (0 e 1)
    if df['Churn'].dtype == 'O': # Se for texto
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. O famoso Get Dummies (Transforma categorias em colunas numéricas)
    # O drop_first=True evita redundância (ex: se não é Homem, é Mulher)
    df_final = pd.get_dummies(df, drop_first=True)
    
    return df_final