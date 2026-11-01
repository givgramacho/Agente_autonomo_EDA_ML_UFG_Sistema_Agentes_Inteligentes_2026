# src/app.py
import os
import streamlit as st
from agent import build_agent, ask_agent, load_csv
from glob import glob
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

st.set_page_config(page_title="Agente Engenheiro de Dados e de Machine Learning", layout="wide", page_icon="📊")
st.title("📊 Agente Engenheiro de Dados e de Machine Learning")

# Inicializar session_state para persistir o agente e LLM
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
    
# Configuração no sidebar
st.sidebar.header("⚙️ Configurações do Agente")
provider = st.sidebar.selectbox(
    "🌐 Provedor LLM",
    ["OpenAI", "Google Gemini", "Hugging Face", "Ollama (local)"]
)

api_key_input = st.sidebar.text_input("🔑 API Key / Token", type="password")

# Modelos por provedor
if provider == "OpenAI":
    model_input = st.sidebar.selectbox(
        "🤖 Modelo OpenAI",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = model_input

elif provider == "Google Gemini":
    model_input = st.sidebar.selectbox(
        "🤖 Modelo Gemini",
        ["gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025", "gemini-flash-latest"]
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = model_input

elif provider == "Hugging Face":
    model_input = st.sidebar.selectbox(
        "🤗 Modelo HuggingFace",
        [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-2-9b-it"
        ]
    )
    if api_key_input:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key_input
        os.environ["HF_TOKEN"] = api_key_input
    os.environ["LLM_PROVIDER"] = "huggingface"
    os.environ["LLM_MODEL"] = model_input

elif provider == "Ollama (local)":
    model_input = st.sidebar.text_input("Modelo Ollama", value="llama3.1:8b")
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = model_input



# Upload do CSV
uploaded = st.file_uploader("📂 Envie um arquivo CSV", type=["csv"])

if uploaded:
    csv_path = os.path.join("data", uploaded.name)
    os.makedirs("data", exist_ok=True)
    
    # Só recarrega se for um arquivo diferente
    if st.session_state.current_file != uploaded.name:
        with open(csv_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Arquivo CSV salvo em: `{csv_path}`")
        load_csv(csv_path)
        # build_agent() agora retorna (agent, llm)
        agent, llm = build_agent()
        st.session_state.agent = agent
        st.session_state.llm = llm
        st.session_state.current_file = uploaded.name
    
    agent = st.session_state.agent
    llm = st.session_state.llm
else:
    st.info("Envie um arquivo CSV para começar.")
    agent = None
    llm = None

# Quick Actions / Exemplos de perguntas
st.markdown("##### 💡 Exemplos rápidos para testar o agente:")
ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)

with ex_col1:
    if st.button("📊 Schema & EDA"):
        st.session_state.example_query = "Mostre o schema, tipos de dados e resumo do dataset."
with ex_col2:
    if st.button("📈 Regressão Linear"):
        st.session_state.example_query = "Execute uma regressão linear para prever a última coluna numérica usando as outras features, mostre o R2, MSE e a equação."
with ex_col3:
    if st.button("🎯 Classificação"):
        st.session_state.example_query = "Treine um modelo de classificação (Decision Tree ou KNN) para a coluna Class ou target e mostre a acurácia e matriz de confusão."
with ex_col4:
    if st.button("🔍 Clusterização"):
        st.session_state.example_query = "Avalie o número ideal de clusters usando o método do cotovelo e coeficiente de silhueta, e depois plote os agrupamentos K-Means."

# Caixa de pergunta
default_query = st.session_state.get("example_query", "")
query = st.text_input("💬 Pergunta para o agente", 
    value=default_query,
    placeholder="Exemplo: Faça uma regressão linear, treine um classificador ou avalie clusters k-means...")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔎 Perguntar"):
        if not agent:
            st.warning("Envie um CSV antes de perguntar.")
        else:
            with st.spinner("Agente analisando e executando tarefas de ML/EDA..."):
                ans = ask_agent(agent, query, csv_path=csv_path, llm=llm)
            st.subheader("📥 Resposta do agente")
            st.write(ans)

            # Mostrar gráficos se a resposta mencionar termos visuais ou de ML
            ml_visual_terms = ["gráfico", "plot", "histogram", "boxplot", "scatter", "distribuição", "regressão", "matriz de confusão", "árvore", "silhueta", "cotovelo", "clusters", "pca", "kmeans"]
            if any(keyword in ans.lower() for keyword in ml_visual_terms):
                plots = sorted(glob("plots/*.png"), reverse=True)
                if plots:
                    st.image(plots[0], caption=os.path.basename(plots[0]))

with col2:
    if st.button("📝 Gerar Conclusão Final"):
        if not agent:
            st.warning("Envie um CSV antes de gerar conclusões.")
        else:
            with st.spinner("Gerando conclusão técnica a partir das análises e modelos..."):
                ans = ask_agent(agent, "Quais conclusões você obteve a partir das análises estatísticas e modelos de Machine Learning executados?", csv_path=csv_path, llm=llm)
            st.subheader("📊 Conclusão Final")
            st.write(ans)

            # Mostrar gráficos relevantes da análise
            plots = sorted(glob("plots/*.png"), reverse=True)[:3]
            if plots:
                st.image(plots, caption=[os.path.basename(p) for p in plots])