import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo
import gc

# Configuração da página
st.set_page_config(page_title="Análise de Clusters de Comportamento Online", layout="wide")

# Título
st.title("📊 Análise de Clusters de Comportamento Online")
st.markdown("""
Análise de agrupamento de clientes baseada no dataset
[Online Shoppers Purchase Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
""")

# Sidebar
st.sidebar.header("Configurações da Análise")
max_clusters = st.sidebar.slider("Número máximo de clusters:", 2, 15, 10)
random_state = 42
variancia_minima = st.sidebar.slider("Variância mínima PCA:", 0.7, 0.95, 0.9, 0.05)

# Seleção de variáveis
opcao_variaveis = st.sidebar.selectbox(
    "Conjunto de variáveis:",
    ["Todas as variáveis", "Apenas variáveis de navegação", "Selecionar manualmente"],
    index=1,
)

if opcao_variaveis == "Selecionar manualmente":
    todas_variaveis = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
        "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
        "VisitorType", "Weekend"
    ]
    variaveis_para_cluster = st.sidebar.multiselect(
        "Selecione variáveis:", todas_variaveis,
        default=["Administrative", "Informational", "ProductRelated"]
    )
else:
    variaveis_para_cluster = None

# Variáveis para visualização
variaveis_visualizacao = st.sidebar.multiselect(
    "Variáveis para análise gráfica:",
    ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
     "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"],
    default=["Administrative", "BounceRates", "ExitRates"],
)

# Configurações visuais
tema_graficos = st.sidebar.selectbox("Tema:", ["whitegrid", "darkgrid", "white", "dark", "ticks"], index=0)
tamanho_fonte = st.sidebar.slider("Fonte:", 10, 18, 12)
sns.set_style(tema_graficos)
plt.rcParams["font.size"] = tamanho_fonte

# Carregamento e otimização de memória
@st.cache_data
def load_and_optimize():
    try:
        dataset = fetch_ucirepo(id=468)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        # Downcast numéricos para float32 e int32
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')

        # Mapear categorias com inteiros pequenos
        meses = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "June":6,
                 "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
        visitantes = {"Returning_Visitor":1, "New_Visitor":2, "Other":3}

        df["Month"] = df["Month"].map(meses).astype('int8')
        df["VisitorType"] = df["VisitorType"].map(visitantes).astype('int8')
        # Revenue é booleano, manter como bool (pouco espaço)
        df["Revenue"] = df["Revenue"].astype('bool')

        # Converter colunas restantes object para category (se houver)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return None

df = load_and_optimize()
if df is None:
    st.stop()

# Recarregar dados
if st.sidebar.button("🔄 Recarregar Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(f"""
**Configurações:**
- Máx. clusters: {max_clusters}
- Variância PCA: {int(variancia_minima*100)}%
""")

# --- Corpo principal ---
st.header("📈 Análise Descritiva")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Registros:** {df.shape[0]}")
    st.write(f"**Variáveis:** {df.shape[1]}")
    st.write(f"**Ausentes:** {df.isna().sum().sum()}")
with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    df["Revenue"].value_counts().plot(kind='bar', ax=ax, color=['skyblue','lightcoral'])
    ax.set_title("Distribuição de Revenue")
    st.pyplot(fig)
    plt.close(fig)

# --- PCA ---
st.header("🔍 Análise de Componentes Principais (PCA)")

# Usar float32 para escalonamento
scaler_pca = StandardScaler()
df_pca_scaled = scaler_pca.fit_transform(df.select_dtypes(include=np.number).astype('float32'))
pca = PCA(svd_solver='randomized', random_state=random_state)
pca.fit(df_pca_scaled)

var_acum = np.cumsum(pca.explained_variance_ratio_)
n_componentes_opt = int(np.argmax(var_acum >= variancia_minima) + 1)
st.write(f"**{n_componentes_opt} componentes** explicam {variancia_minima*100:.0f}% da variância.")

# Scree plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
num_comp = np.arange(1, pca.n_components_+1)
ax1.plot(num_comp, pca.explained_variance_, 'o-', color='blue', markersize=4)
ax1.axvline(n_componentes_opt, color='red', linestyle='--', label=f'{n_componentes_opt} comp.')
ax1.legend()
ax2.plot(num_comp, var_acum, 'o-', color='blue', markersize=4)
ax2.axhline(variancia_minima, color='red', linestyle='--', label=f'{int(variancia_minima*100)}%')
ax2.axvline(n_componentes_opt, color='green', linestyle='--')
ax2.legend()
st.pyplot(fig)
plt.close(fig)

# Liberar memória intermediária
del df_pca_scaled
gc.collect()

# --- Clusterização ---
st.header("🎯 Clusterização (KMeans)")

# Selecionar colunas para cluster
if opcao_variaveis == "Todas as variáveis":
    cols_cluster = [c for c in df.columns if c != 'Revenue']
elif opcao_variaveis == "Apenas variáveis de navegação":
    cols_cluster = ["Administrative", "Administrative_Duration", "Informational",
                    "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]
else:  # manual
    cols_cluster = [c for c in variaveis_para_cluster if c in df.columns] if variaveis_para_cluster else []
    if not cols_cluster:
        cols_cluster = ["Administrative", "Administrative_Duration", "Informational",
                        "Informational_Duration", "ProductRelated", "ProductRelated_Duration"]
        st.warning("Nenhuma variável válida selecionada. Usando variáveis de navegação.")

df_cluster = df[cols_cluster].astype('float32')

@st.cache_data
def get_scaled_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler

df_cluster_scaled, scaler_cluster = get_scaled_data(df_cluster)

# Silhouette dados originais
silhuette_scores_original = []
for n in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=5)
    kmeans.fit(df_cluster_scaled)
    silhuette_scores_original.append(silhouette_score(df_cluster_scaled, kmeans.labels_))

st.subheader("Score de Silhouette - Dados Originais")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(2, max_clusters+1), silhuette_scores_original, 'o-')
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Silhouette com PCA
max_possible = df_cluster_scaled.shape[1]
n_pca = min(n_componentes_opt, max_possible)
st.write(f"**PCA com {n_pca} componentes** (máx: {max_possible})")

pca_cluster = PCA(n_components=n_pca, svd_solver='randomized', random_state=random_state)
df_pca_reduced = pca_cluster.fit_transform(df_cluster_scaled)
var_exp = pca_cluster.explained_variance_ratio_.sum()
st.write(f"Variância explicada: {var_exp:.2%}")

silhuette_scores_pca = []
for n in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=5)
    kmeans.fit(df_pca_reduced)
    silhuette_scores_pca.append(silhouette_score(df_pca_reduced, kmeans.labels_))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(2, max_clusters+1), silhuette_scores_pca, 'o-', color='green')
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

best_n_clusters = int(np.argmax(silhuette_scores_original) + 2)

# Seleção final de k
opcoes_k = list(range(2, max_clusters+1))
selected_k = st.sidebar.selectbox("Número de clusters (k):", opcoes_k, index=opcoes_k.index(3) if 3 in opcoes_k else 1)
selected_k = int(selected_k)

kmeans_final = KMeans(n_clusters=selected_k, random_state=random_state, n_init=5)
kmeans_final.fit(df_cluster_scaled)

df_resultado = df.copy()
df_resultado["Cluster"] = pd.Categorical.from_codes(kmeans_final.labels_, [f"Grupo_{i}" for i in range(selected_k)])

st.info(f"**Clusterização com k = {selected_k}** (recomendado: {best_n_clusters})")

# --- Análise dos Clusters ---
st.header("👥 Análise dos Clusters")

# Distribuição
fig, ax = plt.subplots(figsize=(10,6))
proporcao = df_resultado["Cluster"].value_counts(normalize=True) * 100
bars = proporcao.plot(kind='bar', color='lightblue', ax=ax)
for i, v in enumerate(proporcao):
    ax.text(i, v+0.5, f"{v:.1f}%", ha='center', va='bottom')
ax.set_title(f"Proporção dos Clusters (k={selected_k})")
st.pyplot(fig)
plt.close(fig)

# Visualização por variáveis
if variaveis_visualizacao:
    st.subheader("Análise por Variáveis Selecionadas")
    for var in variaveis_visualizacao:
        if var in df_resultado.columns:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(data=df_resultado, x="Cluster", y=var, hue="Revenue",
                        errorbar=None, palette="coolwarm", ax=ax)
            ax.set_title(f"Média de {var} por Cluster e Revenue")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

# Revenue por cluster
st.subheader("Revenue por Cluster")
fig, ax = plt.subplots(figsize=(10,6))
revenue_by_cluster = df_resultado.groupby(["Cluster", "Revenue"]).size().unstack()
revenue_by_cluster.plot(kind='bar', ax=ax, color=['lightcoral','lightgreen'])
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge")
st.pyplot(fig)
plt.close(fig)

# Estatísticas
cols_stats = ["Administrative", "Informational", "ProductRelated", "BounceRates", "ExitRates", "Revenue"]
cols_stats = [c for c in cols_stats if c in df_resultado.columns]
if cols_stats:
    st.subheader("Estatísticas por Cluster")
    cluster_stats = df_resultado.groupby("Cluster")[cols_stats].mean().round(3)
    st.dataframe(cluster_stats)

# Insights
st.header("💡 Insights e Conclusões")
st.markdown("""
**Principais observações:**
- Usuários que acessam frequentemente páginas administrativas e informativas tendem a apresentar maior taxa de conversão.
- Altos valores de taxa de rejeição (bounce rate) e taxa de saída (exit rate) indicam menor engajamento e menor propensão à compra.
- Recomenda-se realizar o estudo com 3 agrupamentos, pois utilizar menos do que isso tende apenas a separar os “heavy users” do restante do público.
""")

# Limpeza final
del df_cluster_scaled, df_pca_reduced
gc.collect()