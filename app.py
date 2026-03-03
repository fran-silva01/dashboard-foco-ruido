import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard de Foco e Ruído", layout="wide")

# --- CSS PARA MELHORAR O VISUAL (Cards e Cores) ---
st.markdown("""
    <style>
    /* Estiliza o quadrado em volta da métrica */
    [data-testid="stMetric"] {
        background-color: #f0f8ff; /* Azul bem clarinho (AliceBlue) */
        border: 2px solid #005b96;  /* Borda azul sólida */
        padding: 15px;
        border-radius: 10px;       /* Cantos levemente arredondados */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Sombra suave */
    }
    
    /* Muda a cor dos itens selecionados no Multiselect para Azul */
    span[data-baseweb="tag"] {
        background-color: #005b96 !important;
    }
    /* Muda a cor do Slider para Azul */
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background-color: #005b96 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/background_noise_focus_dataset.csv')
    return df

df = load_data()

# --- SIDEBAR (WIDGETS) ---
st.sidebar.title("Filtros de Análise")
st.sidebar.markdown("Use os controles abaixo para filtrar os dados em tempo real.")

role = st.sidebar.multiselect(
    "Selecione o Papel (Role):",
    options=df['role'].unique(),
    default=df['role'].unique()
)

volume_range = st.sidebar.slider(
    "Nível de Volume do Ruído:",
    min_value=int(df['noise_volume_level'].min()),
    max_value=int(df['noise_volume_level'].max()),
    value=(0, 10)
)

df_filtered = df[
    (df['role'].isin(role)) &
    (df['noise_volume_level'].between(volume_range[0], volume_range[1]))
]

# --- CABEÇALHO ---
st.title("Dashboard:")
st.image("images/banner.png", caption="Estudo sobre produtividade em diferentes condições acústicas", use_container_width=True)

# Métricas com estilo de card
st.markdown("### Indicadores Principais")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Participantes", len(df_filtered))
col2.metric("Média da Qualidade", f"{df_filtered['task_completion_quality'].mean():.2f}")
col3.metric("Fadiga Média", f"{df_filtered['mental_fatigue_after_task'].mean():.2f}")

st.markdown("<br>", unsafe_allow_html=True) # Espaçador

# --- GRÁFICOS INTERATIVOS ---
# --- CÓDIGO INTERATIVO (Insights com Plotly) ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1.Impacto do Ruído na Qualidade e Fadiga")

    # 1. Preparação dos dados agrupados por tipo de ruído
    dados_ruido = df_filtered.groupby('background_noise_type')[['task_completion_quality', 'mental_fatigue_after_task']].mean().sort_values(by='task_completion_quality', ascending=False).reset_index()

    # 2. Criando o Gráfico de Barras Interativo
    fig_ruido = px.bar(
        dados_ruido,
        x='background_noise_type',
        y='task_completion_quality',
        color='task_completion_quality',
        color_discrete_sequence=px.colors.sequential.Blues_r, 
        text_auto='.2f',
        title='Impacto do Ruído na Qualidade da Tarefa e Fadiga Mental',
        labels={
            'background_noise_type': "Tipo de Ruído",
            'task_completion_quality': "Média da Qualidade",
            'mental_fatigue_after_task': "Média de Fadiga"
        },
        hover_data={'mental_fatigue_after_task': ': .2f'}
    )
    
    # 3. Adicionando a anotação do "Paradoxo" 
    fig_ruido.add_annotation(
        x=dados_ruido['background_noise_type'].iloc[0], # Aponta para a primeira barra (maior qualidade)
        y=dados_ruido['task_completion_quality'].iloc[0],
        text="<b>Paradoxo:</b> Ruído de tráfego mantém<br>alta qualidade, mas gera maior fadiga.",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-40,
        bgcolor="yellow",
        opacity=0.8
)

    # 4. Ajustes finais de layout para ficar profissional e clean
    fig_ruido.update_layout(
        plot_bgcolor='white',
        yaxis_range=[0, 8],
        coloraxis_showscale=False # Remove a barra lateral de cores para um look mais clean
)

    # Exibindo no Streamlit
    st.plotly_chart(fig_ruido, use_container_width=True)

with col_right:
    st.subheader("2. Distribuição de Foco por Papel")
    fig2 = px.box(
        df_filtered, 
        x='role', 
        y='perceived_focus_score', 
        color='role',
        color_discrete_sequence=px.colors.qualitative.Safe,
        points='all'
    )
    fig2.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig2, use_container_width=True)

# --- GRÁFICOS ESTÁTICOS ---
st.markdown("---")
st.subheader("Análises Adicionais (Estatísticas)")

c1, c2 = st.columns(2)

with c1:
    st.write("**Matriz de Correlação**")
    corr = df_filtered.select_dtypes(include='number').drop(columns=['participant_id'], errors='ignore').corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool))
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', mask=mask, ax=ax_corr, cbar_kws={'shrink': .8})
    st.pyplot(fig_corr)

with c2:
    st.write("**Impacto do Volume no Score de Foco**")
    fig_line, ax_line = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df_filtered, x='noise_volume_level', y='perceived_focus_score', marker='o', color='#005b96', ax=ax_line)
    ax_line.set_facecolor('white')
    ax_line.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_line)


# --- CÓDIGO INTERATIVO (Insight 1 com Plotly) ---
st.subheader("Análise Explanatória de Idade (Interativo)")

media_idade = df_filtered['age'].mean()
mediana_idade = df_filtered['age'].median()
contagem_papeis = df_filtered['role'].value_counts(normalize=True) * 100
texto_anotacao_plotly = "<b>Distribuição por Papel:</b><br>" + "<br>".join([f"{k}: {v:.1f}%" for k, v in contagem_papeis.items()])

fig_age = px.histogram(
    df_filtered, 
    x='age', 
    color='role', 
    barmode='stack',
    nbins=20,
    title="Análise Explanatória: Distribuição de Idade por Categoria",
    labels={'age': "Idade (Anos)", 'count': "Contagem", 'role': "Papel"},
    color_discrete_sequence=px.colors.sequential.Blues_r,
    template="simple_white"
)

fig_age.add_vline(x=media_idade, line_dash="dash", line_color="red", 
                  annotation_text=f"Média: {media_idade:.1f}", annotation_position="top left")

fig_age.add_vline(x=mediana_idade, line_dash="solid", line_color="blue", 
                  annotation_text=f"Mediana: {mediana_idade:.1f}", annotation_position="top right")

fig_age.add_annotation(
    x=0.95, y=0.90,
    xref='paper', yref='paper',
    text=texto_anotacao_plotly,
    showarrow=False,
    align='left',
    bgcolor="rgba(255, 255, 255, 0.8)",
    bordercolor="gray",
    borderwidth=1
)

fig_age.update_layout(yaxis_title="Contagem de Participantes", hovermode='x unified')
st.plotly_chart(fig_age, use_container_width=True)

