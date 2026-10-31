# src/agent.py
import os
import logging
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_core.prompts import MessagesPlaceholder

# Importar tools base e set_dataframe de tools.py
from tools import (
    schema_tool, dataset_info_tool, missing_tool, describe_tool, histogram_tool,
    set_dataframe
)

# Importar tools adicionais de tools_refactored.py
from tools_refactored import (
    boxplot_tool, scatter_tool, correlation_tool, outliers_tool,
    clustering_tool, time_trend_tool, frequency_tool, crosstab_tool,
    central_tendency_tool, variability_tool, range_tool,
    class_balance_tool, conclusion_tool
)

# Importar ferramentas de Machine Learning de tools_ml.py
from tools_ml import (
    regression_linear_tool, regression_polynomial_tool, predict_regression_tool,
    classification_model_tool, decision_tree_plot_tool,
    cluster_evaluation_tool, cluster_kmeans_plot_tool, pca_reduction_tool
)

from memory_store import init_memory
from langsmith_setup import get_langsmith_client
from utils import logger


# Empacotar tools com descrições inteligentes e orientações claras
TOOLS = [
    Tool(name="schema", func=lambda q: schema_tool(""), description="Retorna schema (colunas e tipos). USE para estrutura - NÃO gera gráficos."),
    Tool(name="dataset_info", func=lambda q: dataset_info_tool(""), description="Retorna informações COMPLETAS do dataset: shape, tipos, memória, missing values, estatísticas. USE para perguntas sobre estrutura, contagem, tamanho - NÃO gera gráficos."),
    Tool(name="missing", func=lambda q: missing_tool(""), description="Retorna colunas com valores ausentes. USE para perguntas sobre dados faltantes - NÃO gera gráficos."),
    Tool(name="describe", func=lambda q: describe_tool(q), description="Estatísticas descritivas (média, mediana, desvio). USE para perguntas sobre médias, medianas, totais - NÃO gera gráficos."),
    Tool(name="histogram", func=lambda q: histogram_tool(q), description="Cria histograma. USE APENAS quando usuário pedir 'mostre', 'visualize', 'distribuição' ou análise visual."),
    Tool(name="boxplot", func=lambda q: boxplot_tool(q), description="Cria boxplot. USE APENAS para comparações de distribuições ou quando usuário pedir visualização."),
    Tool(name="scatter", func=lambda q: scatter_tool(q), description="Scatter plot. USE APENAS para análise de correlação visual ou quando pedido explícito."),
    Tool(name="correlation", func=lambda q: correlation_tool(""), description="Calcula matriz de correlação entre variáveis numéricas. USE para análise de relacionamentos - NÃO gera gráficos."),
    Tool(name="outliers", func=lambda q: outliers_tool(q), description="Detecta outliers. USE para análise de valores extremos - NÃO gera gráficos."),
    Tool(name="clustering", func=lambda q: clustering_tool(q), description="K-means clustering simples. USE para segmentação de dados tabular."),
    Tool(name="time_trend", func=lambda q: time_trend_tool(q), description="Análise temporal. USE para séries temporais - NÃO gera gráficos."),
    Tool(name="frequency", func=lambda q: frequency_tool(q), description="Valores mais frequentes. USE para contagens e rankings - NÃO gera gráficos."),
    Tool(name="crosstab", func=lambda q: crosstab_tool(q), description="Tabela cruzada. USE para análise categórica - NÃO gera gráficos."),
    Tool(name="central_tendency", func=lambda q: central_tendency_tool(q), description="Média, mediana, moda. USE para perguntas sobre tendências centrais - NÃO gera gráficos."),
    Tool(name="variability", func=lambda q: variability_tool(q), description="Variância, desvio padrão, CV. USE para perguntas sobre dispersão - NÃO gera gráficos."),
    Tool(name="range", func=lambda q: range_tool(q), description="Min e max de uma coluna. USE para limites e extremos - NÃO gera gráficos."),
    Tool(name="class_balance", func=lambda q: class_balance_tool(""), description="Balanceamento de classes (coluna 'Class'). USE para análise de distribuição - NÃO gera gráficos."),
    Tool(name="conclusion", func=lambda q: conclusion_tool(""), description="Gera conclusão baseada na memória. USE para resumos - NÃO gera gráficos."),
    
    # Ferramentas de Machine Learning (Regressão, Classificação, Clusterização e PCA)
    Tool(name="regression_linear", func=lambda q: regression_linear_tool(q), description="Regressão Linear (Simples ou Múltipla). Calcula MSE, RMSE, MAE, R², equação da reta e gera gráfico da reta ajustada ou real vs previsto. Params: target=y, features=x1|x2, test_size=0.2."),
    Tool(name="regression_polynomial", func=lambda q: regression_polynomial_tool(q), description="Regressão Polinomial com curva ajustada e métricas MSE, R². Params: target=y, feature=x, degree=2."),
    Tool(name="predict_regression", func=lambda q: predict_regression_tool(q), description="Faz predição ou estimativa de valor para regressão dado um valor de entrada. Params: target=y, feature=x, value=100.0."),
    Tool(name="classification_model", func=lambda q: classification_model_tool(q), description="Classificação supervisionada (KNN, Árvore de Decisão CART, Random Forest, Regressão Logística). Avalia acurácia, precisão, recall, F1, ROC-AUC e gera Matriz de Confusão gráfica. Params: target=Class, algorithm=knn|tree|rf|logistic, max_depth=5, n_neighbors=5."),
    Tool(name="decision_tree_plot", func=lambda q: decision_tree_plot_tool(q), description="Plota diagrama da Árvore de Decisão com nós, divisão e critérios Gini/Entropia. Params: target=Class, max_depth=3, criterion=gini."),
    Tool(name="cluster_evaluation", func=lambda q: cluster_evaluation_tool(q), description="Avaliação não supervisionada de clusters: Método do Cotovelo (Inércia/WCSS), Silhueta, Davies-Bouldin e Calinski-Harabasz. Gera 4 gráficos comparativos e recomenda o melhor k. Params: max_k=8, columns=col1|col2."),
    Tool(name="cluster_kmeans_plot", func=lambda q: cluster_kmeans_plot_tool(q), description="Clusterização K-Means final com centros e visualização gráfica 2D dos agrupamentos e centroides. Params: k=3, x=col1, y=col2."),
    Tool(name="pca_reduction", func=lambda q: pca_reduction_tool(q), description="Redução de Dimensionalidade com PCA (Análise de Componentes Principais). Gera Scree Plot com variância explicada acumulada e projeção 2D (PC1 x PC2). Params: hue_column=Class.")
]


def load_csv(path: str):
    """Carrega CSV e define como DataFrame global."""
    try:
        df = pd.read_csv(path)
        set_dataframe(df)
        logger.info(f"CSV loaded successfully: {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {path}: {e}")
        raise


def detect_user_intent(question: str) -> str:
    """
    Detecta a intenção do usuário baseado em palavras-chave.
    Retorna: 'simple_stats', 'visualization', 'machine_learning', ou 'analysis'
    """
    question_lower = question.lower()
    
    # Palavras-chave para tarefas de Machine Learning
    ml_keywords = [
        "regressão", "regressao", "previsão", "previsao", "predição", "predicao", "estimar", "prever",
        "classificação", "classificacao", "classifique", "supervisionado", "knn", "árvore", "arvore", "decision tree",
        "acurácia", "acuracia", "matriz de confusão", "matriz de confusao", "f1-score", "recall", "precisão", "precisao",
        "cluster", "clusters", "clusterização", "clusterizacao", "agrupamento", "cotovelo", "elbow", "silhueta",
        "silhouette", "k-means", "kmeans", "não supervisionado", "nao supervisionado", "pca", "componentes principais"
    ]
    
    # Palavras-chave para estatísticas simples (sem gráficos)
    simple_keywords = [
        "quantos", "quantas", "quantidade", "total", "contagem",
        "média", "mediana", "moda", "desvio", "valor",
        "mínimo", "máximo", "range", "tamanho", "linhas"
    ]
    
    # Palavras-chave para visualização (com gráficos)
    visualization_keywords = [
        "mostre", "visualize", "gráfico", "plot", "distribuição",
        "compare", "veja", "disposição", "padrão visual"
    ]
    
    # Verificar intenção
    if any(keyword in question_lower for keyword in ml_keywords):
        return "machine_learning"
    elif any(keyword in question_lower for keyword in simple_keywords):
        return "simple_stats"
    elif any(keyword in question_lower for keyword in visualization_keywords):
        return "visualization"
    else:
        return "analysis"


# Cache para evitar visualizações repetidas
_visualization_cache = {}

def should_generate_visualization(tool_name: str, params: str, user_intent: str) -> bool:
    """
    Valida se deve gerar visualização baseado em cache e intenção.
    Retorna True se deve gerar, False se usar cache ou não gerar.
    """
    # Para estatísticas simples, NUNCA gerar visualizações
    if user_intent == "simple_stats":
        logger.info(f"Blocking visualization for simple_stats intent: {tool_name}")
        return False
    
    # Criar chave cache
    cache_key = f"{tool_name}:{hash(params)}"
    
    # Verificar cache
    if cache_key in _visualization_cache:
        logger.info(f"Using cached visualization: {tool_name}")
        return False
    
    # Para visualizações e machine learning, permitir mas marcar no cache
    if user_intent in ["visualization", "machine_learning"]:
        _visualization_cache[cache_key] = True
        logger.info(f"Allowing visualization for {user_intent} intent: {tool_name}")
        return True
    
    # Para análise, decidir baseado na tool
    visualization_tools = ["histogram", "boxplot", "scatter", "regression_linear", "regression_polynomial", "classification_model", "decision_tree_plot", "cluster_evaluation", "cluster_kmeans_plot", "pca_reduction"]
    if tool_name in visualization_tools:
        _visualization_cache[cache_key] = True
        logger.info(f"Allowing visualization for analysis intent: {tool_name}")
        return True
    
    return False


def build_agent():
    """Constrói o agente com memória e LLM, suportando OpenAI, Gemini e Ollama.
    
    Returns:
        tuple: (agent, llm) - Retorna o agente e a instância do LLM
    """
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    logger.info(f"Building agent with provider={provider}, model={model}")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API Key da OpenAI não definida.")
        llm = ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API Key do Google Gemini não definida.")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, api_key=api_key)

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model=model, temperature=0)

    elif provider in ["huggingface", "hf"]:
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("Token da HuggingFace (HUGGINGFACEHUB_API_TOKEN ou HF_TOKEN) não definido.")
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
            endpoint = HuggingFaceEndpoint(
                repo_id=model,
                temperature=0.01,
                huggingfacehub_api_token=api_key
            )
            llm = ChatHuggingFace(llm=endpoint)
        except ImportError:
            from langchain_community.llms import HuggingFaceHub
            llm = HuggingFaceHub(
                repo_id=model,
                model_kwargs={"temperature": 0.01},
                huggingfacehub_api_token=api_key
            )

    else:
        raise ValueError(f"Provedor {provider} não suportado.")

    # Memória persistente
    mems = init_memory()
    memory = mems.get("buffer")

    # Prompt customizado SUPER REFORÇADO com suporte integral a EDA e Machine Learning
    prefix = """Você é um Agente Autônomo Especialista em Engenharia de Dados, Análise Exploratória e Machine Learning (Aprendizado de Máquina).

Você é capaz de executar com excelência 3 grandes categorias de tarefas de Machine Learning:
1. 📈 PREVISÃO / REGRESSÃO:
   - Regressão Linear Simples e Múltipla (`regression_linear`)
   - Regressão Polinomial com curvas de ajuste (`regression_polynomial`)
   - Predição e inferência de novos valores (`predict_regression`)
   - Métricas: MSE, RMSE, MAE, R² e Equação da Reta

2. 🎯 CLASSIFICAÇÃO (APRENDIZADO SUPERVISIONADO):
   - Treinamento e avaliação com KNN, Árvores de Decisão CART, Random Forest e Regressão Logística (`classification_model`)
   - Diagrama visual da árvore com nós e divisões (`decision_tree_plot`)
   - Métricas: Acurácia, Precisão, Recall, F1-Score, Matriz de Confusão e ROC-AUC

3. 🔍 CLUSTERIZAÇÃO E REDUÇÃO (APRENDIZADO NÃO SUPERVISIONADO):
   - Avaliação diagnóstica de clusters: Método do Cotovelo (Inércia/WCSS), Silhueta, Davies-Bouldin e Calinski-Harabasz (`cluster_evaluation`)
   - K-Means com plot 2D de agrupamentos e centroides (`cluster_kmeans_plot`)
   - Redução de Dimensionalidade com PCA (Scree Plot e projeção 2D) (`pca_reduction`)

🚨 DIRETRIZES DE EXECUÇÃO:
- Se o usuário pedir regressão, use as ferramentas de regressão (`regression_linear`, `regression_polynomial`, `predict_regression`).
- Se o usuário pedir classificação, use `classification_model` ou `decision_tree_plot`.
- Se o usuário pedir agrupamento, número ideal de clusters ou K-means, use `cluster_evaluation` ou `cluster_kmeans_plot`.
- Se o usuário pedir PCA ou redução de dimensionalidade, use `pca_reduction`.
- Para estatísticas simples (quantos, média, total), use `dataset_info`, `describe` ou `central_tendency` (NÃO gere gráficos desnecessários).

Você tem acesso às seguintes ferramentas:"""

    suffix = """SEMPRE comece com Thought, depois Action, depois Action Input.

Chat History: {chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        max_iterations=8,
        handle_parsing_errors=True,
        max_execution_time=120,
        early_stopping_method="generate",
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix,
        }
    )
    
    logger.info("Agent built successfully with memory")
    # Retornar agente e LLM como tupla
    return agent, llm


def ask_agent(agent, question: str, csv_path: str = None, llm=None):
    """Executa uma pergunta ao agente com detecção de intenção inteligente.
    
    Args:
        agent: O agente executor
        question: Pergunta em linguagem natural
        csv_path: Caminho opcional para CSV
        llm: Instância do LLM (necessário para conclusões detalhadas)
    """
    try:
        if csv_path:
            load_csv(csv_path)
        
        # Detectar intenção do usuário para orientar o agente
        user_intent = detect_user_intent(question)
        logger.info(f"User intent detected: {user_intent}")
        
        # Enriquecer a pergunta com contexto de intenção
        enhanced_question = question
        if user_intent == "simple_stats":
            enhanced_question = f"{question} (FOCO: estatísticas simples, NÃO gere gráficos)"
        elif user_intent == "visualization":
            enhanced_question = f"{question} (FOCO: análise visual, USE gráficos)"
        elif user_intent == "machine_learning":
            enhanced_question = f"{question} (FOCO: Machine Learning - Regressão, Classificação ou Clusterização. Treine o modelo solicitado, calcule as métricas e gere o gráfico correspondente)"
        else:
            enhanced_question = f"{question} (FOCO: análise completa, decida sobre gráficos)"
        
        logger.info(f"Processing enhanced question: {enhanced_question[:100]}...")
        
        # Executar pergunta com memória e contexto
        response = agent.run(enhanced_question)

        # Se a pergunta for de conclusão, melhora o resumo com análise detalhada
        if "conclusão" in question.lower():
            history = agent.memory.load_memory_variables({})
            context = str(history.get("chat_history", ""))
            
            # Usa LLM passado como parâmetro ou tenta acessar do agente
            if llm is None:
                try:
                    llm = agent.agent.llm_chain.llm
                except:
                    # Fallback: retorna resposta simples se não conseguir acessar LLM
                    logger.warning("Could not access LLM for enhanced conclusion")
                    return response
            
            conclusion_prompt = f"""Você é um Engenheiro de Machine Learning e Analista de Dados experiente.

Baseado nas análises realizadas no dataset, gere um relatório técnico completo incluindo:

## 📊 Histórico de Análises:
{context}

## 📋 Resposta do Agente:
{response}

## 🎯 Sua Tarefa:
Gere uma conclusão técnica profissional com:
1. **Resumo Executivo** das principais descobertas
2. **Insights Técnicos** sobre padrões identificados
3. **Recomendações** para próximos passos de análise
4. **Observações Estatísticas** relevantes

Formate em Markdown."""

            try:
                enhanced_response = llm.predict(conclusion_prompt)
                return enhanced_response
            except Exception as e:
                logger.error(f"Error generating enhanced conclusion: {e}")
                return response

        return response

    except Exception as e:
        logger.error(f"Error in ask_agent: {e}")
        return f"Erro ao processar pergunta: {str(e)}"
