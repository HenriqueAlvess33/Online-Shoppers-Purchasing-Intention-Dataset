import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# Inicializa session_state para armazenar buffers
if "fig_buffers" not in st.session_state:
    st.session_state.fig_buffers = {}


def plotagem_em_png(numero_da_imagem, fig, tamanho=800):
    """
    Converte a figura matplotlib em PNG, armazena no session_state e exibe.
    """
    if st.session_state.fig_buffers is None:
        st.session_state.fig_buffers = {}
    buff = io.BytesIO()
    fig.savefig(buff, format="png", bbox_inches="tight")
    buff.seek(0)
    st.session_state.fig_buffers[numero_da_imagem] = buff
    st.image(buff, width=tamanho)


# Configuração da página
st.set_page_config(
    page_title="Análise de Clusters de Comportamento Online", layout="wide"
)

# Título da aplicação
st.title("📊 Análise de Clusters de Comportamento Online")
st.markdown(
    """
    Análise de agrupamento de clientes baseada no dataset
    [Online Shoppers Purchase Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
    """
)

# Sidebar com controles
st.sidebar.header("Configurações da Análise")

# Controles principais
max_clusters = st.sidebar.slider(
    "Número máximo de clusters para testar:",
    min_value=2,
    max_value=15,
    value=10,
    help="Define quantas soluções de agrupamento serão testadas",
)

random_state = 42  # fixo para reprodutibilidade

variancia_minima = st.sidebar.slider(
    "Variância mínima explicada pelo PCA:",
    min_value=0.7,
    max_value=0.95,
    value=0.9,
    step=0.05,
    help="Percentual mínimo de variância a ser explicado pelos componentes principais",
)

# Seleção das variáveis para clusterização
st.sidebar.subheader("Variáveis para Clusterização")
opcao_variaveis = st.sidebar.selectbox(
    "Escolha o conjunto de variáveis:",
    options=[
        "Todas as variáveis",
        "Apenas variáveis de navegação",
        "Selecionar manualmente",
    ],
    index=1,
    help="Define quais variáveis serão usadas para formar os clusters",
)

if opcao_variaveis == "Selecionar manualmente":
    todas_variaveis = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend",
    ]
    variaveis_para_cluster = st.sidebar.multiselect(
        "Selecione as variáveis para o agrupamento:",
        options=todas_variaveis,
        default=["Administrative", "Informational", "ProductRelated"],
        help="As variáveis selecionadas serão usadas para formar os clusters",
    )
else:
    variaveis_para_cluster = None  # será definido depois

# Variáveis para visualização nos gráficos (separadas)
st.sidebar.subheader("Variáveis para Visualização")
variaveis_visualizacao = st.sidebar.multiselect(
    "Selecione as variáveis para análise gráfica:",
    options=[
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ],
    default=["Administrative", "BounceRates", "ExitRates"],
    help="Escolha quais variáveis visualizar nos gráficos de análise",
)

# Configurações de visualização
st.sidebar.subheader("Configurações de Visualização")
tema_graficos = st.sidebar.selectbox(
    "Tema dos gráficos:",
    options=["whitegrid", "darkgrid", "white", "dark", "ticks"],
    index=0,
)
tamanho_fonte = st.sidebar.slider(
    "Tamanho da fonte:", min_value=10, max_value=18, value=12
)

# Aplicar configurações
sns.set_style(tema_graficos)
plt.rcParams["font.size"] = tamanho_fonte


# Carregamento dos dados (cache)
@st.cache_data
def load_data():
    try:
        dataset = fetch_ucirepo(id=468)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        return df, dataset
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None


# Botão para recarregar dados
if st.sidebar.button("🔄 Recarregar Dados"):
    st.cache_data.clear()
    st.rerun()

# Informações adicionais na sidebar
st.sidebar.markdown("---")
st.sidebar.header("Sobre a Análise")
st.sidebar.info(
    f"""
    **Configurações atuais:**
    - Máx. clusters: {max_clusters}
    - Variância PCA: {int(variancia_minima*100)}%
    - Random state: {random_state}
    """
)

# --------------------------------------------------------------------
# Corpo principal da aplicação
# --------------------------------------------------------------------

try:
    df, dataset_info = load_data()
    if df is None:
        st.stop()

    # Pré-processamento das variáveis categóricas
    numeracao_meses = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "June": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    tipo_de_visitante = {"Returning_Visitor": 1, "New_Visitor": 2, "Other": 3}

    df_processed = df.copy()
    df_processed["Month"] = df_processed["Month"].map(numeracao_meses)
    df_processed["VisitorType"] = df_processed["VisitorType"].map(tipo_de_visitante)

    # -----------------------------------------------------------------
    # Análise Descritiva
    # -----------------------------------------------------------------
    st.header("📈 Análise Descritiva")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informações do Dataset")
        st.write(f"**Número de registros:** {df.shape[0]}")
        st.write(f"**Número de variáveis:** {df.shape[1]}")
        st.write(f"**Valores ausentes:** {df.isna().sum().sum()}")

    with col2:
        st.subheader("Distribuição de Revenue")
        fig, ax = plt.subplots(figsize=(6, 4))
        revenue_counts = df["Revenue"].value_counts()
        revenue_counts.plot(kind="bar", ax=ax, color=["skyblue", "lightcoral"])
        ax.set_title("Distribuição de Revenue")
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Frequência")
        plotagem_em_png(1, fig, tamanho=600)  # ID 1

    # -----------------------------------------------------------------
    # Análise de Componentes Principais (PCA) sobre todas as variáveis
    # -----------------------------------------------------------------
    st.header("🔍 Análise de Componentes Principais (PCA)")

    # Padroniza todos os dados para a PCA
    scaler_pca = StandardScaler()
    df_pca_scaled = scaler_pca.fit_transform(df_processed)
    pca = PCA()
    pca.fit(df_pca_scaled)

    # Determina o número de componentes que atingem a variância mínima
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    n_componentes_opt = int(
        np.argmax(var_acum >= variancia_minima) + 1
    )  # converte para int

    st.write(
        f"**Para explicar {variancia_minima*100:.0f}% da variância, são necessários {n_componentes_opt} componentes.**"
    )

    # Scree Plot
    st.subheader("Scree Plot - Variância Explicada")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    num_comp = np.arange(1, pca.n_components_ + 1)

    ax1.plot(
        num_comp, pca.explained_variance_, "o-", linewidth=2, color="blue", markersize=4
    )
    ax1.set_title("Variância Total por Componente")
    ax1.set_xlabel("Número de Componentes")
    ax1.set_ylabel("Variância Explicada")
    ax1.axvline(
        x=n_componentes_opt,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"{n_componentes_opt} comp.",
    )
    ax1.legend()

    ax2.plot(
        num_comp,
        pca.explained_variance_.cumsum(),
        "o-",
        linewidth=2,
        color="blue",
        markersize=4,
    )
    ax2.set_title("Variância Acumulada")
    ax2.set_xlabel("Número de Componentes")
    ax2.set_ylabel("Variância Acumulada")
    ax2.axhline(
        y=variancia_minima,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"{variancia_minima*100:.0f}%",
    )
    ax2.axvline(x=n_componentes_opt, color="green", linestyle="--", linewidth=1)
    ax2.legend()

    plt.tight_layout()
    plotagem_em_png(2, fig, tamanho=800)  # ID 2

    # -----------------------------------------------------------------
    # Clusterização
    # -----------------------------------------------------------------
    st.header("🎯 Clusterização (KMeans)")

    # Definir quais variáveis serão usadas para agrupamento
    if opcao_variaveis == "Todas as variáveis":
        colunas_cluster = df_processed.columns.tolist()
        if "Revenue" in colunas_cluster:
            colunas_cluster.remove("Revenue")
        df_cluster = df_processed[colunas_cluster].copy()
    elif opcao_variaveis == "Apenas variáveis de navegação":
        colunas_navegacao = [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
        ]
        df_cluster = df_processed[colunas_navegacao].copy()
    else:  # Selecionar manualmente
        if not variaveis_para_cluster:
            st.warning(
                "Nenhuma variável selecionada para clusterização. Usando variáveis de navegação padrão."
            )
            colunas_navegacao = [
                "Administrative",
                "Administrative_Duration",
                "Informational",
                "Informational_Duration",
                "ProductRelated",
                "ProductRelated_Duration",
            ]
            df_cluster = df_processed[colunas_navegacao].copy()
        else:
            colunas_existentes = [
                c for c in variaveis_para_cluster if c in df_processed.columns
            ]
            if len(colunas_existentes) == 0:
                st.error(
                    "Nenhuma variável válida selecionada. Usando variáveis de navegação."
                )
                colunas_navegacao = [
                    "Administrative",
                    "Administrative_Duration",
                    "Informational",
                    "Informational_Duration",
                    "ProductRelated",
                    "ProductRelated_Duration",
                ]
                df_cluster = df_processed[colunas_navegacao].copy()
            else:
                df_cluster = df_processed[colunas_existentes].copy()

    # Padronização dos dados para clusterização (cache)
    @st.cache_data
    def get_scaled_data(df_cluster):
        scaler_cluster = StandardScaler()
        return scaler_cluster.fit_transform(df_cluster)

    df_cluster_scaled = get_scaled_data(df_cluster)

    # -----------------------------------------------------------------
    # Silhouette: dados originais (padronizados)
    # -----------------------------------------------------------------
    silhuette_scores_original = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(df_cluster_scaled)
        silhuette_scores_original.append(
            silhouette_score(df_cluster_scaled, kmeans.labels_)
        )

    # Gráfico do silhouette (dados originais)
    st.subheader("Score de Silhouette - Dados Originais")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(2, max_clusters + 1),
        silhuette_scores_original,
        "o-",
        linewidth=2,
        markersize=8,
    )
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Score de Silhouette")
    ax.set_title("Score de Silhouette por Número de Clusters (Dados Originais)")
    ax.grid(True, alpha=0.3)
    plotagem_em_png(3, fig, tamanho=800)  # ID 3

    # -----------------------------------------------------------------
    # Silhouette: dados transformados pelo PCA (sobre o conjunto de cluster)
    # -----------------------------------------------------------------
    st.subheader("Score de Silhouette - Dados Transformados por PCA")

    max_possible_components = df_cluster_scaled.shape[1]
    n_components_pca = int(
        min(n_componentes_opt, max_possible_components)
    )  # converte para int

    st.write(
        f"**PCA aplicada com {n_components_pca} componentes** (máximo possível: {max_possible_components})"
    )

    pca_cluster = PCA(n_components=n_components_pca)
    df_pca_reduced = pca_cluster.fit_transform(df_cluster_scaled)

    var_exp = pca_cluster.explained_variance_ratio_.sum()
    st.write(
        f"Variância total explicada com {n_components_pca} componentes: {var_exp:.2%}"
    )

    silhuette_scores_pca = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(df_pca_reduced)
        silhuette_scores_pca.append(silhouette_score(df_pca_reduced, kmeans.labels_))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(2, max_clusters + 1),
        silhuette_scores_pca,
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
    )
    ax.set_xlabel("Número de Clusters")
    ax.set_ylabel("Score de Silhouette")
    ax.set_title(f"Score de Silhouette com PCA ({n_components_pca} componentes)")
    ax.grid(True, alpha=0.3)
    plotagem_em_png(4, fig, tamanho=800)  # ID 4

    # Número recomendado (convertido para int nativo)
    best_n_clusters = int(np.argmax(silhuette_scores_original) + 2)

    # Sidebar: seleção do número de clusters
    st.sidebar.subheader("Seleção de Clusters")
    opcoes_k = list(range(2, max_clusters + 1))

    selected_k = st.sidebar.selectbox(
        "Número de clusters (k) para análise:",
        options=opcoes_k,
        index=1,
        help="Escolha o número de clusters para a segmentação final",
    )
    # Garantir que selected_k seja int (selectbox já retorna int, mas por segurança)
    selected_k = int(selected_k)

    # Aplicar KMeans com o k selecionado
    kmeans_selected = KMeans(
        n_clusters=selected_k, random_state=random_state, n_init=10
    )
    kmeans_selected.fit(df_cluster_scaled)

    # Adicionar os rótulos dos clusters ao DataFrame original
    df_resultado = df_processed.copy()
    nomes_grupos = [f"Grupo_{i}" for i in range(selected_k)]
    df_resultado["Cluster"] = pd.Categorical.from_codes(
        kmeans_selected.labels_, categories=nomes_grupos
    )

    st.info(
        f"**Clusterização ativa com k = {selected_k}** (recomendado: {best_n_clusters})"
    )

    # -----------------------------------------------------------------
    # Análise dos Clusters
    # -----------------------------------------------------------------
    st.header("👥 Análise dos Clusters")

    # Proporção dos clusters
    st.subheader("Distribuição dos Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    proporcao = df_resultado["Cluster"].value_counts(normalize=True) * 100
    bars = proporcao.plot(kind="bar", color="lightblue", ax=ax)

    # Adicionar rótulos com percentuais
    for i, valor in enumerate(proporcao):
        ax.text(i, valor + 0.5, f"{valor:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title(f"Proporção dos Clusters (k = {selected_k})")
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Proporção (%)")
    plt.xticks(rotation=45)
    plotagem_em_png(5, fig, tamanho=800)  # ID 5

    # Análise por variáveis selecionadas
    if variaveis_visualizacao:
        st.subheader("Análise por Variáveis Selecionadas")
        plot_id = 6
        for var in variaveis_visualizacao:
            if var in df_resultado.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Barplot com hue=Revenue
                sns.barplot(
                    data=df_resultado,
                    x="Cluster",
                    y=var,
                    hue="Revenue",
                    errorbar=None,
                    palette="coolwarm",
                    ax=ax,
                )
                ax.set_title(f"Média de {var} por Cluster e Revenue")
                ax.set_ylabel(f"Média de {var}")
                plt.xticks(rotation=45)

                # Adicionar rótulos com os valores das médias
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=9)

                plotagem_em_png(plot_id, fig, tamanho=800)
                plot_id += 1

    # Revenue por cluster
    st.subheader("Revenue por Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    revenue_by_cluster = df_resultado.groupby(["Cluster", "Revenue"]).size().unstack()
    revenue_by_cluster.plot(kind="bar", ax=ax, color=["lightcoral", "lightgreen"])
    ax.set_title("Distribuição de Revenue por Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Número de Usuários")
    ax.legend(title="Revenue", loc="upper right")
    plt.xticks(rotation=45)

    # Adicionar rótulos com as contagens
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=9)

    plotagem_em_png(plot_id, fig, tamanho=800)  # ID final
    plot_id += 1

    # Estatísticas por cluster (médias)
    colunas_estatisticas = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "BounceRates",
        "ExitRates",
        "Revenue",
    ]
    colunas_existentes = [c for c in colunas_estatisticas if c in df_resultado.columns]
    if colunas_existentes:
        st.subheader("Estatísticas por Cluster")
        cluster_stats = (
            df_resultado.groupby("Cluster")[colunas_existentes].mean().round(3)
        )
        st.dataframe(cluster_stats)

    # Insights
    st.header("💡 Insights e Conclusões")
    st.markdown(
        """
        **Principais observações:**
        - Usuários que acessam frequentemente páginas administrativas e informativas tendem a apresentar maior taxa de conversão.
        - Altos valores de taxa de rejeição (bounce rate) e taxa de saída (exit rate) indicam menor engajamento e menor propensão à compra.
        - Recomenda-se realizar o estudo com 3 agrupamentos, pois utilizar menos do que isso tende apenas a separar os “heavy users” do restante do público.
        """
    )

except Exception as e:
    st.error(f"Erro na análise: {e}")
    st.stop()
