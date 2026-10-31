# 📊 Agente Autônomo de EDA & Aprendizado de Máquina
### 🏛️ Universidade Federal de Goiás (UFG)
**Programa de Pós-Graduação em Sistemas de Agentes Inteligentes**  
**Disciplina:** Aprendizado de Máquina  
**Repositório Oficial:** [https://github.com/givgramacho/Agente_autonomo_EDA_ML_UFG_Sistema_Agentes_Inteligentes_2026.git](https://github.com/givgramacho/Agente_autonomo_EDA_ML_UFG_Sistema_Agentes_Inteligentes_2026.git)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seu-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Agente inteligente com memória persistente capaz de analisar qualquer arquivo CSV, responder perguntas de Regressão, Classificação, Clusterização e gerar visualizações automáticas.**

---

## 🎯 Sobre o Projeto

Sistema de Machine Learning e Análise Exploratória de Dados desenvolvido para a **Universidade Federal de Goiás (UFG)** utilizando **Agentes Autônomos** com **LangChain** e **LLMs** (OpenAI GPT-4, Google Gemini ou Ollama). O agente possui **memória conversacional**, permitindo referências a análises e modelos treinados anteriormente.

### 📋 Funcionalidades

- ✅ **Ferramentas de Machine Learning Completo**:
  - **Previsão / Regressão**: Linear Simples/Múltipla, Polinomial, Métricas (MSE, RMSE, R²), Equação da Reta e Predição.
  - **Classificação Supervisionada**: KNN, Árvores de Decisão CART, Random Forest, Matriz de Confusão, Acurácia, F1-Score e Diagrama de Árvore.
  - **Clusterização & Não Supervisionado**: Avaliação de Clusters (Cotovelo/WCSS, Silhueta, Davies-Bouldin, Calinski-Harabasz), K-Means 2D e PCA.
- ✅ **18 Ferramentas de Análise Exploratória (EDA)** (histograma, correlação, outliers, etc.)
- ✅ **Memória Persistente** - Lembra de conversas e modelos executados
- ✅ **Suporte Multi-LLM** - OpenAI, Gemini, Ollama
- ✅ **Interface Web** - Streamlit interativo com botões rápidos
- ✅ **Visualizações Automáticas** - Gráficos salvos automaticamente
- ✅ **Conclusão Inteligente** - Resumo técnico e executivo baseado em todas as análises

---

## 🚀 Acesso Rápido

### 🌐 Aplicação Online (Deploy)

👉 **[Acessar Agente EDA Online](https://seu-app.streamlit.app)** _(substitua após deploy)_

### 💻 Instalação Local

#### Pré-requisitos

- Python 3.10 ou superior
- Git
- Conta Google (para API Gemini) ou OpenAI

---

## 📥 Instalação

### 1️⃣ Clonar o Repositório
```bash
git clone https://github.com/givgramacho/Agente_autonomo_EDA_ML_UFG_Sistema_Agentes_Inteligentes_2026.git
cd Agente_autonomo_EDA_ML_UFG_Sistema_Agentes_Inteligentes_2026
```

### 2️⃣ Criar Ambiente Virtual

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# .env

# LLM Provider (escolha: openai, gemini, ou ollama)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-exp

# API Keys (adicione apenas a que você vai usar)
GOOGLE_API_KEY=sua_chave_google_aqui
OPENAI_API_KEY=sua_chave_openai_aqui

# LangSmith (opcional - para debugging)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=sua_chave_langsmith_aqui
```

#### 🔑 Como Obter API Keys

**Google Gemini (Recomendado - Gratuito):**

1. Acesse [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Clique em "Get API Key"
3. Copie a chave e cole no `.env`

**OpenAI (Pago):**

1. Acesse [OpenAI Platform](https://platform.openai.com/api-keys)
2. Crie uma nova API Key
3. Copie e cole no `.env`

**Ollama (Local - Gratuito):**

1. Instale [Ollama](https://ollama.ai/download)
2. Execute: `ollama pull llama2`
3. Configure `LLM_PROVIDER=ollama` no `.env`

---

## ▶️ Como Executar

### Rodar a Aplicação Streamlit

```bash
streamlit run src/app.py
```

A aplicação abrirá automaticamente em: **http://localhost:8501**

### Testar o Sistema de Memória

```bash
python test_memoria.py
```

---

## 📖 Como Usar

### 1. Fazer Upload do CSV

- Clique em **"📁 Upload CSV"** na barra lateral
- Selecione seu arquivo CSV (ex: `creditcard.csv`)
- Aguarde o carregamento

### 2. Fazer Perguntas

Exemplos de perguntas que você pode fazer:

```
✅ "Mostre o schema do dataset"
✅ "Quantas linhas e colunas existem?"
✅ "Crie um histograma da coluna Amount"
✅ "Mostre a correlação entre todas as variáveis"
✅ "Detecte outliers na coluna Time usando método IQR"
✅ "Qual é a média e mediana da coluna Amount?"
✅ "Faça um clustering com 3 grupos nas colunas V1, V2, V3"
✅ "Mostre o balanceamento das classes"
✅ "Execute uma regressão linear para prever Preco com base na Area: qual o R2, MSE e a equação da reta?" ← REGRESSÃO
✅ "Faça uma regressão polinomial de grau 2 e estime o valor para uma área de 150m²" ← REGRESSÃO POLINOMIAL
✅ "Treine um modelo de classificação (Decision Tree ou KNN) para prever a classe e mostre a matriz de confusão e acurácia" ← CLASSIFICAÇÃO SUPERVISIONADA
✅ "Avalie o número ótimo de clusters pelo método do cotovelo e coeficiente de silhueta" ← APRENDIZADO NÃO SUPERVISIONADO
✅ "Aplique PCA para reduzir a dimensionalidade em 2D e plote o Scree Plot de variância explicada" ← PCA
✅ "Qual foi minha primeira pergunta?" ← TESTE DE MEMÓRIA
```

### 3. Gerar Conclusão Final

- Clique no botão **"📝 Gerar Conclusão Final"**
- O agente irá:
  - Revisar TODO o histórico de análises e modelos treinados
  - Gerar um relatório executivo completo
  - Incluir insights técnicos, métricas de performance e recomendações

---

## 🏗️ Estrutura do Projeto

```
agentes_engenheiro_dados/
├── src/
│   ├── app.py                  # Interface Streamlit com botões rápidos de ML
│   ├── agent.py                # Agente LangChain com memória e orquestração de ML/EDA
│   ├── tools.py                # Ferramentas base (schema, info, histogram)
│   ├── tools_refactored.py     # Ferramentas adicionais (boxplot, scatter, outliers...)
│   ├── tools_ml.py             # Ferramentas de Machine Learning (Regressão, Classificação, Clusters, PCA)
│   ├── memory_store.py         # Configuração de memória
│   ├── utils.py                # Funções utilitárias
│   ├── langgraph_flow.json     # Fluxograma da arquitetura dos agentes
│   └── langsmith_setup.py      # Integração LangSmith
├── data/                       # Datasets de exemplo
├── plots/                      # Gráficos gerados (auto-criado)
├── test_memoria.py             # Script de teste de memória
├── test_ml_capabilities.py     # Script de validação das ferramentas de ML
├── requirements.txt            # Dependências Python
├── .env                        # Variáveis de ambiente (NÃO COMMITAR)
├── .gitignore                  # Arquivos ignorados
└── README.md                   # Documentação do projeto
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia              | Versão   | Descrição                   |
| ----------------------- | --------- | ----------------------------- |
| **Python**        | 3.10+     | Linguagem principal           |
| **LangChain**     | 0.3.12    | Framework de agentes          |
| **Streamlit**     | 1.41.1    | Interface web                 |
| **Pandas**        | 2.2.3     | Manipulação de dados        |
| **Matplotlib**    | 3.10.0    | Visualizações               |
| **Seaborn**       | 0.13.2    | Gráficos estatísticos       |
| **Scikit-learn**  | 1.6.1     | Machine Learning (clustering) |
| **Google Gemini** | 2.0 Flash | LLM principal                 |
| **OpenAI GPT-4**  | Mini      | LLM alternativo               |

---

## 🚢 Deploy no Streamlit Cloud

### Passo 1: Preparar Repositório GitHub

```bash
# Criar repositório no GitHub
git init
git add .
git commit -m "Initial commit - Agente EDA"
git branch -M main
git remote add origin https://github.com/seu-usuario/agentes_engenheiro_dados.git
git push -u origin main
```

### Passo 2: Deploy no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Faça login com GitHub
3. Clique em **"New app"**
4. Configure:
   - **Repository:** `seu-usuario/agentes_engenheiro_dados`
   - **Branch:** `main`
   - **Main file path:** `src/app.py`
5. Em **"Advanced settings"** → **"Secrets"**, adicione:

```toml
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.0-flash-exp"
GOOGLE_API_KEY = "sua_chave_aqui"
LANGSMITH_TRACING = "false"
```

6. Clique em **"Deploy!"**
7. Aguarde 2-3 minutos
8. Sua URL será: `https://seu-app.streamlit.app`

---

## 📊 Ferramentas Disponíveis (18 Tools)

| #  | Ferramenta           | Descrição               | Exemplo                    |
| -- | -------------------- | ------------------------- | -------------------------- |
| 1  | `schema`           | Schema do dataset         | "Mostre as colunas"        |
| 2  | `dataset_info`     | Informações completas   | "Info do dataset"          |
| 3  | `missing`          | Valores ausentes          | "Valores nulos?"           |
| 4  | `describe`         | Estatísticas descritivas | "Descreva os dados"        |
| 5  | `histogram`        | Histograma                | "Histograma de Amount"     |
| 6  | `boxplot`          | Boxplot                   | "Boxplot de V1"            |
| 7  | `scatter`          | Dispersão                | "Scatter de V1 vs V2"      |
| 8  | `correlation`      | Correlação              | "Matriz de correlação"   |
| 9  | `outliers`         | Outliers                  | "Outliers em Amount"       |
| 10 | `clustering`       | K-means                   | "3 clusters em V1,V2,V3"   |
| 11 | `time_trend`       | Tendências temporais     | "Tendência de Time"       |
| 12 | `frequency`        | Frequências              | "Valores frequentes"       |
| 13 | `crosstab`         | Tabela cruzada            | "Crosstab Class vs Amount" |
| 14 | `central_tendency` | Média/mediana/moda       | "Média de Amount"         |
| 15 | `variability`      | Desvio padrão/variância | "Variância de Time"       |
| 16 | `range`            | Min/Max                   | "Range de Amount"          |
| 17 | `class_balance`    | Balanceamento             | "Classes balanceadas?"     |
| 18 | `conclusion`       | Conclusão final          | "Gere conclusão"          |

---

## 🧠 Sistema de Memória

O agente possui **memória conversacional persistente** usando `ConversationBufferMemory` do LangChain.

### Como Funciona:

```python
# Em agent.py
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Prompt com histórico
suffix = """
Chat History: {chat_history}  ← Histórico completo
Question: {input}
"""
```

### Testando a Memória:

```
Pergunta 1: "Mostre o schema"
Pergunta 2: "Crie histograma de Amount"
Pergunta 3: "Qual foi minha primeira pergunta?"  ← TESTE

Resposta: "Sua primeira pergunta foi 'Mostre o schema'"
```

---

## 🧪 Testes

### Teste Automático de Memória

```bash
python test_memoria.py
```

**Saída Esperada:**

```
🧪 TESTE DO SISTEMA DE MEMÓRIA
================================================================================
📦 Construindo agente com memória...
✅ Agente construído com sucesso!
================================================================================
📂 Carregando CSV: data/1_01_file_creditcard.csv
✅ CSV carregado com sucesso!
================================================================================
📝 TESTE 1: Primeira pergunta
   Pergunta: Mostre o schema do dataset
   Resposta: O dataset possui 31 colunas: Time (float64), V1 (float64)...
✅ Teste 1 passou!
================================================================================
🧠 TESTE 3: MEMÓRIA - Qual foi minha primeira pergunta?
   Resposta: Sua primeira pergunta foi "Mostre o schema do dataset"
✅ MEMÓRIA FUNCIONANDO! O agente lembrou da primeira pergunta!
```

---

## 🔒 Segurança

### ⚠️ NUNCA COMMITAR CHAVES DE API

O arquivo `.gitignore` já está configurado para ignorar:

```gitignore
.env
*.env
.env.local
secrets.toml
```

### Verificar Antes de Commitar:

```bash
git status  # Verifique que .env NÃO aparece
git diff    # Revise mudanças antes de commit
```

---

## 📝 Exemplos de Uso

### Exemplo 1: Análise Básica

```python
# Upload: creditcard.csv

Pergunta: "Mostre informações do dataset"
Resposta:
{
  "shape": {"rows": 284807, "columns": 31},
  "columns": ["Time", "V1", ..., "Class"],
  "missing_values": {},
  "duplicates": 0
}
```

### Exemplo 2: Visualização

```python
Pergunta: "Crie um histograma da coluna Amount com 50 bins"
Resposta:
"Histograma criado para 'Amount'"
Plot: plots/hist-Amount-20250108-223045.png
Stats: {"mean": 88.35, "median": 22.0, "std": 250.12}
```

### Exemplo 3: Detecção de Outliers

```python
Pergunta: "Detecte outliers na coluna Amount usando método IQR"
Resposta:
{
  "method": "IQR",
  "outliers_count": 7741,
  "percentage": 2.72%,
  "bounds": {"lower": -39.5, "upper": 85.5}
}
```

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-ferramenta`)
3. Commit suas mudanças (`git commit -m 'Add nova ferramenta'`)
4. Push para a branch (`git push origin feature/nova-ferramenta`)
5. Abra um Pull Request

---

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.

---

## 👥 Autores

- **Seu Nome** - [GitHub](https://github.com/seu-usuario)
- **Projeto Acadêmico** - Agentes Autônomos IA 2025

---

## 📚 Referências

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## 📞 Suporte

**Problemas Comuns:**

### 1. Erro: "No dataframe loaded"

**Solução:** Faça upload do CSV antes de perguntar

### 2. Erro: "API Key não definida"

**Solução:** Configure o arquivo `.env` corretamente

### 3. Erro: "OutputParserException"

**Solução:** O agente tentará automaticamente até 8 vezes

### 4. Deploy não funciona

**Solução:** Verifique os secrets no Streamlit Cloud

---

## ⭐ Agradecimentos

Se este projeto foi útil, deixe uma ⭐ no GitHub!

**Link do Projeto:** https://github.com/seu-usuario/agentes_engenheiro_dados

**Link da Aplicação:** https://seu-app.streamlit.app

---

<div align="center">
  <strong>Desenvolvido com ❤️ Givanildo Gramacho - I2A2 Curso de agentes inteligentes 2025</strong>
</div>
