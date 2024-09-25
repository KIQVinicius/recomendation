import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# Definir sua chave de API Alpha Vantage
api_key = 'J7X1VBF0HXTJB9I8'

# Suponha que o DataFrame seja 'df' e você queira remover os outliers de uma coluna específica 'coluna_alvo'
def remover_outliers_iqr(df, coluna_alvo):
    Q1 = df[coluna_alvo].quantile(0.25)
    Q3 = df[coluna_alvo].quantile(0.75)
    IQR = Q3 - Q1

    # Limites inferior e superior
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # Filtrar os dados dentro dos limites
    df_sem_outliers = df[(df[coluna_alvo] >= limite_inferior) & (df[coluna_alvo] <= limite_superior)]
    return df_sem_outliers

df = pd.read_csv('/content/credit_features_formatado.csv')

df.head()

df = df.drop(columns=['Unnamed: 0'])

df.info()

df.head()

df = remover_outliers_iqr(df, 'RENDA_PESSOAL_MENSAL')

def cluster_investors(profiles, num_clusters=3):
    # Escolher apenas as colunas que fazem sentido para a clusterização
    X = profiles[['IDADE', 'RENDA_PESSOAL_MENSAL']]

    # Aplicar o KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    profiles['cluster'] = kmeans.labels_

    # Exibir resultados
    plt.scatter(X['IDADE'], X['RENDA_PESSOAL_MENSAL'], c=profiles['cluster'], cmap='viridis')
    plt.title('Clusterização de Perfis de Investidores')
    plt.xlabel('Idade')
    plt.ylabel('Renda')
    plt.show()

    return profiles

# Aplicar clusterização
clustered_profiles = cluster_investors(df)
print(clustered_profiles[['ID_CLIENTE', 'IDADE', 'RENDA_PESSOAL_MENSAL', 'cluster']])

def get_investment_statistics(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

    # Calcular retorno médio e volatilidade
    data['daily_return'] = data['4. close'].pct_change()  # Calcular retorno diário
    avg_return = data['daily_return'].mean() * 252  # Retorno anualizado
    volatility = data['daily_return'].std() * np.sqrt(252)  # Volatilidade anualizada

    return avg_return, volatility

# Exemplo: Obter estatísticas reais para as ações
investments_real_data = {}
symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMZN', 'JNJ', 'PG']

for symbol in symbols:
    avg_return, volatility = get_investment_statistics(symbol, api_key)
    investments_real_data[symbol] = {'avg_return': avg_return, 'volatility': volatility}

print(investments_real_data)

def recommend_investment_by_cluster(profiles, investments_real_data):
    recommendations = []
    for _, profile in profiles.iterrows():
        cluster = profile['cluster']

        if cluster == 0:  # Cluster conservador -> risco baixo, volatilidade baixa
            low_risk_investments = {k: v for k, v in investments_real_data.items() if v['volatility'] < 0.15}
            if low_risk_investments:  # Verifica se há investimentos disponíveis
                rec = max(low_risk_investments, key=lambda k: low_risk_investments[k]['avg_return'])
            else:
                rec = 'No low-risk investments available'  # Mensagem padrão

        elif cluster == 1:  # Cluster moderado -> risco médio
            medium_risk_investments = {k: v for k, v in investments_real_data.items() if 0.15 <= v['volatility'] < 0.25}
            if medium_risk_investments:  # Verifica se há investimentos disponíveis
                rec = max(medium_risk_investments, key=lambda k: medium_risk_investments[k]['avg_return'])
            else:
                rec = 'No medium-risk investments available'  # Mensagem padrão

        else:  # Cluster arriscado -> risco alto, alta volatilidade
            high_risk_investments = {k: v for k, v in investments_real_data.items() if v['volatility'] >= 0.25}
            if high_risk_investments:  # Verifica se há investimentos disponíveis
                rec = max(high_risk_investments, key=lambda k: high_risk_investments[k]['avg_return'])
            else:
                rec = 'No high-risk investments available'  # Mensagem padrão

        recommendations.append(rec)

    profiles['recommendation'] = recommendations
    return profiles

# Exibir perfis recomendados com clusters e recomendações
def plot_recommendations(profiles):
    fig, ax = plt.subplots()

    # Plotar cada cluster com cor distinta
    colors = ['green', 'orange', 'red']
    for cluster_id in profiles['cluster'].unique():
        cluster_data = profiles[profiles['cluster'] == cluster_id]
        ax.scatter(cluster_data['IDADE'], cluster_data['RENDA_PESSOAL_MENSAL'],
                   label=f'Cluster {cluster_id}',
                   color=colors[cluster_id], alpha=0.6, edgecolors='w', s=100)

    ax.set_xlabel('Idade')
    ax.set_ylabel('Renda')
    ax.set_title('Clusters de Investidores e Recomendações')
    ax.legend()

    plt.show()

# Plotar os clusters e as recomendações
plot_recommendations(df)

