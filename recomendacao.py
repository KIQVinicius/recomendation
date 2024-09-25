import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from alpha_vantage.timeseries import TimeSeries
import streamlit as st

# Definir sua chave de API Alpha Vantage
API_KEY = 'DEOCB3ZQJU2OPM1I'

# Função para remover outliers usando o método IQR
def remover_outliers_iqr(df, coluna_alvo):
    Q1 = df[coluna_alvo].quantile(0.25)
    Q3 = df[coluna_alvo].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna_alvo] >= limite_inferior) & (df[coluna_alvo] <= limite_superior)]

# Função para realizar clusterização com KMeans
def cluster_investors(profiles, num_clusters=3):
    X = profiles[['IDADE', 'RENDA_PESSOAL_MENSAL']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    profiles['cluster'] = kmeans.labels_
    return profiles

# Função para obter estatísticas de investimento
def get_investment_statistics(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        data['daily_return'] = data['4. close'].pct_change()
        avg_return = data['daily_return'].mean() * 252
        volatility = data['daily_return'].std() * np.sqrt(252)
        return avg_return, volatility
    except Exception as e:
        print(f"Erro ao obter dados da API para {symbol}: {e}")
        return get_local_data(symbol)

# Função para simular dados locais ou obter dados de uma base de dados local
def get_local_data(symbol):
    local_data = {
        'AAPL': (0.1, 0.2),  # Exemplo: (avg_return, volatility)
        'GOOGL': (0.08, 0.18),
        'TSLA': (0.15, 0.25),
        'AMZN': (0.12, 0.22),
        'JNJ': (0.06, 0.12),
        'PG': (0.04, 0.1),
        'BND': (0.03, 0.05),
        'TLT': (0.05, 0.07),
        'PFF': (0.04, 0.09)
    }
    return local_data.get(symbol, (0, 0))

# Função para recomendar investimentos com base nos clusters
def recommend_investment_by_cluster(profiles, investments_real_data):
    recommendations = []
    risk_levels = []
    for _, profile in profiles.iterrows():
        cluster = profile['cluster']
        if cluster == 0:
            low_risk_investments = {k: v for k, v in investments_real_data.items() if v[1] < 0.15}
            rec = max(low_risk_investments, key=lambda k: low_risk_investments[k][0], default='No low-risk investments available')
            risk_levels.append('Low Risk')
        elif cluster == 1:
            medium_risk_investments = {k: v for k, v in investments_real_data.items() if 0.15 <= v[1] < 0.25}
            rec = max(medium_risk_investments, key=lambda k: medium_risk_investments[k][0], default='No medium-risk investments available')
            risk_levels.append('Medium Risk')
        else:
            high_risk_investments = {k: v for k, v in investments_real_data.items() if v[1] >= 0.25}
            rec = max(high_risk_investments, key=lambda k: high_risk_investments[k][0], default='No high-risk investments available')
            risk_levels.append('High Risk')
        recommendations.append(rec)
    profiles['recommendation'] = recommendations
    profiles['risk_level'] = risk_levels
    return profiles

# Configuração do Streamlit
st.title("Sistema de Recomendação de Investimentos")

# Carregar os dados
df = pd.read_csv('credit_features_formatado.csv')
df = df.drop(columns=['Unnamed: 0'])
df = remover_outliers_iqr(df, 'RENDA_PESSOAL_MENSAL')

# Aplicar a clusterização
df_clustered = cluster_investors(df)

# Obter dados reais dos investimentos
symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'JNJ', 'PG', 'BND', 'TLT', 'PFF']
investments_real_data = {symbol: get_investment_statistics(symbol, API_KEY) for symbol in symbols}

# Aplicar recomendação com base nos clusters
df_with_recommendations = recommend_investment_by_cluster(df_clustered, investments_real_data)

# Filtro para o nível de risco
risk_filter = st.selectbox("Selecione o Nível de Risco", ['Todos', 'Baixo', 'Médio', 'Alto'])

# Filtrar o DataFrame com base no nível de risco selecionado
if risk_filter == 'Baixo':
    df_filtered = df_with_recommendations[df_with_recommendations['risk_level'] == 'Low Risk']
elif risk_filter == 'Médio':
    df_filtered = df_with_recommendations[df_with_recommendations['risk_level'] == 'Medium Risk']
elif risk_filter == 'Alto':
    df_filtered = df_with_recommendations[df_with_recommendations['risk_level'] == 'High Risk']
else:
    df_filtered = df_with_recommendations

# Exibir o DataFrame filtrado
st.subheader("Dados de Investimento")
st.dataframe(df_filtered)

# Lista de IDs de clientes disponíveis (apenas dos clientes filtrados)
client_ids = df_filtered['ID_CLIENTE'].unique().tolist()

# Dropdown com pesquisa para seleção do ID do Cliente
selected_client_id = st.selectbox("Selecione o ID do Cliente:", client_ids)

# Verificar se o ID do usuário foi fornecido
if selected_client_id:
    # Filtrar os dados com base no ID do usuário selecionado
    user_profile = df_filtered[df_filtered['ID_CLIENTE'] == selected_client_id]
    
    if not user_profile.empty:
        # Exibir as recomendações do usuário
        st.header(f"Recomendações para o Cliente ID {selected_client_id}")
        for _, profile in user_profile.iterrows():
            st.write(f"**Recomendação**: {profile['recommendation']} - **Nível de Risco**: {profile['risk_level']} - **Cluster**: {profile['cluster']}")
    else:
        st.warning("ID do Cliente não encontrado.")
else:
    st.info("Por favor, selecione um ID de Cliente para obter recomendações.")
