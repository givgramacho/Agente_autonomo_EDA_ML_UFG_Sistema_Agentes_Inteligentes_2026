# Projeto: Agente Autônomo de EDA & Machine Learning

**Instituição:** Universidade Federal de Goiás (UFG)  
**Programa:** Pós-Graduação em Sistemas de Agentes Inteligentes  
**Disciplina:** Aprendizado de Máquina (Período: 05/09/2026 a 31/10/2026)  
**Repositório GitHub:** [https://github.com/givgramacho/I2A2_agentes_autonomos_DataScience.git](https://github.com/givgramacho/I2A2_agentes_autonomos_DataScience.git)  
**Pasta local:** `@agentes_engenheiro_dados`  
**Referência:** `instrucao_trabalho.md` e `T2-python_scripts`

## Objetivo

Construir um agente genérico capaz de:

- Carregar qualquer arquivo CSV fornecido pelo usuário;
- Responder perguntas analíticas (estatísticas descritivas, distribuição, outliers, correlações, padrões temporais, agrupamentos);
- Gerar gráficos (histograma, scatter, barras, boxplot, séries temporais);
- Produzir **conclusões** a partir das análises;
- Manter **memória** das interações para justificar conclusões e permitir follow-ups;
- Expor uma interface (web) onde o usuário submete perguntas e recebe textos + gráficos;

## Tecnologias / Frameworks

- Python 3.10+
- LangChain (agentes e ferramentas)
- LangGraph (para diagramar fluxos/depêndencias dos agentes)
- LangSmith (tracking / armazenamento de conversas e execuções)
- Streamlit (UI web minimal)
- Pandas, NumPy, Scikit-learn (EDA / clustering), Matplotlib (plots), Seaborn (opcional)
- Vectorstore (Chroma / FAISS) para memória vetorial (opcional)
- GIT + Docker (opcional para deploy)

## Estrutura do projeto (sugestão)

@agentes_engenheiro_dados/

├─ README.md

├─ DOCUMENTO_PROJETO.md # este arquivo

├─ requirements.txt

├─ Dockerfile (opcional)

├─ src/

│ ├─ app.py # Streamlit front-end (UI)

│ ├─ agent.py # orquestrador do agente (LangChain Agent)

│ ├─ tools.py # ferramentas (EDA, plots, clustering, outlier)

│ ├─ langgraph_flow.json # representação do fluxo do agente (LangGraph)

│ ├─ langsmith_setup.py # helper para logs e tracking

│ ├─ memory_store.py # inicializa vectorstore e memória do agente

│ ├─ utils.py # utilitários (parse params, sanitize)

│ └─ tests/

│ └─ test_agent_basic.py

├─ data/

│ └─ example_creditcard.csv # exemplo (credit card fraud)

└─ reports/

└─ Agentes Autônomos – Relatório da Atividade Extra.pdf

## Interface / Como o agente deve funcionar (comportamento)

- Usuário sobe um CSV (ou fornece link) e faz perguntas em linguagem natural.
- O agente decide, com base no prompt/intent, qual **tool** chamar:
  - `schema` → retorna colunas + tipos
  - `missing` → retorna colunas com NaNs
  - `describe` → estatísticas descritivas
  - `histogram` / `scatter` / `bar` / `boxplot` → plota e salva imagem
  - `correlation` → matriz de correlação, heatmap
  - `clustering` → tenta k-means / dbscan e retorna centros / clusters
  - `outliers` → IQR / z-score e lista outliers
- O agente grava as ações/insights na **memória** (vetorial + conversa).
- Quando respondendo “Conclusões”, agrega automaticamente: resumo dos principais sinais detectados (variáveis com maior variação, clusters, anomalias e recomendações).

## Requisitos mínimos de entrega

1. Framework escolhida (descrição).
2. Estrutura da solução (arquitetura) — fluxograma (LangGraph).
3. Pelo menos 4 perguntas e respostas (uma com gráfico).
4. Uma pergunta sobre conclusões e resposta.
5. Código-fonte ou JSON
6. Link para testar o agente (ou instruções para rodar local).
7. Chaves e segredos ocultos.
8. **Capacidades Completas de Machine Learning (Instrução do Trabalho da Disciplina):**
   - **Previsão (Regressão):** Modelos lineares simples, múltiplos e polinomiais, MSE, RMSE, R², equação da reta e predição de novos valores.
   - **Classificação (Supervisionado):** KNN, Árvores de Decisão CART, Random Forest, Regressão Logística, Acurácia, F1-Score, Matriz de Confusão e visualização de árvore.
   - **Agrupamento/Clusterização (Não Supervisionado):** Diagnóstico com Método do Cotovelo (Inércia WCSS), Coeficiente de Silhueta, Índices Davies-Bouldin e Calinski-Harabasz, K-Means e PCA 2D/3D.

## Exemplo de perguntas que o agente precisa responder (valide estes no teste)

### 📊 Análise Exploratória (EDA):
- “Mostre o schema e tipos de dados do CSV.”
- “Crie um histograma da coluna `Amount` com 50 bins.” (gráfico)
- “Quais colunas têm valores ausentes?”
- “Há correlações fortes (>0.7) entre variáveis?”
- “Detecte outliers na coluna `Amount` e me diga se devem ser removidos.”

### 📈 Previsão (Regressão):
- “Treine uma regressão linear para prever a variável target a partir das features numéricas. Qual o R², MSE e a equação da reta?”
- “Ajuste uma regressão polinomial de grau 2 na coluna X e estime a previsão quando X = 120.”

### 🎯 Classificação (Supervisionado):
- “Treine um modelo de classificação (Decision Tree ou KNN) para prever a classe do dataset. Apresente a acurácia, F1-Score e gere o gráfico da Matriz de Confusão.”
- “Mostre o diagrama visual da Árvore de Decisão com regras de divisão e critério Gini.”

### 🔍 Agrupamento / Clusterização (Não Supervisionado):
- “Avalie qual é o melhor número de clusters k usando o método do cotovelo e o coeficiente de silhueta.”
- “Execute o K-Means com k=3 e plote os dados e os centroides.”
- “Aplique PCA para reduzir a dimensionalidade para 2 componentes e plote o Scree Plot de variância explicada.”

### 📝 Conclusões Técnicas:
- “Quais conclusões e recomendações você extrai deste dataset e dos modelos de Machine Learning executados?”

## Observações técnicas

- Prefira ferramentas que retornem **texto curto + JSON** para que a UI consiga renderizar e tomar ações (ex.: mostrar gráficos).
- Para memória: use embedding + vectorstore (Chroma/FAISS). Salve também a sessão no LangSmith para auditoria.
- Não permita que o agente “adivinhe” sem executar ferramentas: sempre que a resposta depender de dados, **exija** a chamada de ferramenta (policy no prompt).

## Passos de execução (local)

1. ` python3 --version`
2. `uv venv --python 3.11`
3. ```
   uv package manager

   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Update PATH to use the new uv version
   export PATH="/Users/$USER/.local/bin:$PATH"
   ```

4. Criar ambiente virtual: `python -m venv .venv && source .venv/bin/activate`
5. Instalar dependências: `pip install -r requirements.txt`
6. Rodar Streamlit: `streamlit run src/app.py`
7. Subir CSV e fazer perguntas.

## Como gerar o PDF final (relatório)

- Gere um arquivo `relatorio.md` (ou Jupyter Notebook) com as seções pedidas.
- Converter para PDF: `pandoc relatorio.md -o "Agentes Autônomos – Relatório da Atividade Extra.pdf"` ou usar `nbconvert` se usar notebook.
- Anexar códigos fonte + link de acesso no e-mail para `challenges@i2a2.academy`. :contentReference[oaicite:3]{index=3}

## Segurança e chaveamento

- Nunca comitar chaves. Use variáveis de ambiente `.env` e `gitignore`.
- Para LangSmith / OpenAI defina: `OPENAI_API_KEY`, `LANGSMITH_API_KEY` no ambiente, e no código leia via `os.getenv`.

---
