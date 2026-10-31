# src/tools_ml.py
"""
Módulo de Ferramentas de Aprendizado de Máquina (ML)
Baseado nos scripts e metodologias da disciplina (T2-python_scripts):
1. Regressão (Linear Simples/Múltipla, Polinomial, Métricas MSE, RMSE, MAE, R², Equação da Reta, Previsões e Gráficos)
2. Classificação Supervisionada (KNN, Árvore de Decisão CART, Random Forest, Logistic Regression, Métricas: Acurácia, Precisão, Recall, F1, Matriz de Confusão, ROC-AUC)
3. Clusterização / Aprendizado Não Supervisionado (K-Means com Método do Cotovelo/Inércia WCSS, Silhueta, Davies-Bouldin, Calinski-Harabasz, Redução PCA 2D/3D com Variância Explicada)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from langchain_core.tools import tool

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import parse_tool_params, get_param, logger, safe_json_convert
from tools import get_dataframe, _save_plot


# ==============================================================================
# 1. FERRAMENTAS DE REGRESSÃO (PREVISÃO CONTÍNUA)
# ==============================================================================

@tool
def regression_linear_tool(params: str) -> str:
    """
    Treina modelo de Regressão Linear Simples ou Múltipla.
    Calcula intercepto, coeficientes, MSE, RMSE, MAE e R².
    Gera gráfico de dispersão com reta ajustada ou gráfico de Observado vs Previsto.
    Params: target=coluna_y, features=col1|col2 (se omitido, usa colunas numéricas), test_size=0.2
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
    
    try:
        params_dict = parse_tool_params(params)
        target = get_param(params_dict, "target", None)
        features_str = get_param(params_dict, "features", None)
        test_size = get_param(params_dict, "test_size", 0.2, float)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not target:
            # Fallback inteligente: última coluna numérica
            if not numeric_cols:
                return json.dumps({"error": "Dataset não possui colunas numéricas para regressão."})
            target = numeric_cols[-1]
            
        if target not in df.columns:
            return json.dumps({"error": f"Coluna target '{target}' não encontrada."})
            
        if features_str:
            features = [c.strip() for c in features_str.split("|") if c.strip() in df.columns]
        else:
            features = [c for c in numeric_cols if c != target]
            
        if not features:
            return json.dumps({"error": "Nenhuma variável preditora (features) numérica válida informada."})
            
        clean_df = df[features + [target]].dropna()
        if len(clean_df) < 5:
            return json.dumps({"error": "Poucos registros válidos após remoção de valores nulos."})
            
        X = clean_df[features].values
        y = clean_df[target].values
        
        # Split treino e teste
        if len(clean_df) > 15:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
            
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        intercept = float(model.intercept_)
        coefficients = {feat: float(coef) for feat, coef in zip(features, model.coef_)}
        
        # Equação formatada
        terms = [f"{coef:+.4f} × {feat}" for feat, coef in zip(features, model.coef_)]
        equation = f"{target} = {intercept:.4f} " + " ".join(terms)
        
        # Gráficos
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(features) == 1:
            feat_name = features[0]
            ax.scatter(X_test[:, 0], y_test, color="navy", alpha=0.6, label="Dados observados (Teste)")
            x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, color="crimson", linewidth=2.5, label="Reta de Regressão Linear")
            ax.set_xlabel(feat_name, fontsize=11, fontweight="bold")
            ax.set_ylabel(target, fontsize=11, fontweight="bold")
            ax.set_title(f"Regressão Linear: {target} vs {feat_name} (R² = {r2:.4f})", fontsize=12, fontweight="bold")
        else:
            # Gráfico de Observado x Previsto
            ax.scatter(y_test, y_pred, color="teal", alpha=0.6, label="Previsões")
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Linha Ideal (y = y_pred)")
            ax.set_xlabel("Valor Real Observado", fontsize=11, fontweight="bold")
            ax.set_ylabel("Valor Previsto pelo Modelo", fontsize=11, fontweight="bold")
            ax.set_title(f"Regressão Linear Múltipla: Real vs Previsto (R² = {r2:.4f})", fontsize=12, fontweight="bold")
            
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plot_path = _save_plot(fig, prefix=f"regressao-linear-{target}")
        
        return json.dumps({
            "task": "Regressão Linear",
            "target": target,
            "features": features,
            "samples": len(clean_df),
            "equation": equation,
            "intercept": intercept,
            "coefficients": coefficients,
            "metrics": {
                "MSE": round(float(mse), 4),
                "RMSE": round(float(rmse), 4),
                "MAE": round(float(mae), 4),
                "R2": round(float(r2), 4)
            },
            "plot_path": plot_path
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Erro em regression_linear_tool: {e}")
        return json.dumps({"error": f"Falha na regressão linear: {str(e)}"})


@tool
def regression_polynomial_tool(params: str) -> str:
    """
    Ajusta uma Regressão Polinomial com grau customizável (degree) e pipeline com StandardScaler.
    Calcula MSE, RMSE, R² e plota a curva ajustada.
    Params: target=coluna_y, feature=coluna_x, degree=2
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        target = get_param(params_dict, "target", None)
        feature = get_param(params_dict, "feature", None)
        degree = get_param(params_dict, "degree", 2, int)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not target and numeric_cols:
            target = numeric_cols[-1]
        if not feature and len(numeric_cols) > 1:
            feature = numeric_cols[0]
            
        if not target or target not in df.columns or not feature or feature not in df.columns:
            return json.dumps({"error": "Necessário definir feature e target numéricos válidos."})
            
        clean_df = df[[feature, target]].dropna()
        X = clean_df[[feature]].values
        y = clean_df[target].values
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression())
        ])
        
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Curva suave para plot
        x_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_curve = pipeline.predict(x_curve)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color="black", s=40, alpha=0.6, label="Dados observados")
        ax.plot(x_curve, y_curve, color="crimson", linewidth=2.5, label=f"Polinômio Grau {degree}")
        ax.set_xlabel(feature, fontsize=11, fontweight="bold")
        ax.set_ylabel(target, fontsize=11, fontweight="bold")
        ax.set_title(f"Regressão Polinomial (Grau {degree}) | R² = {r2:.4f}, MSE = {mse:.4f}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
        plot_path = _save_plot(fig, prefix=f"regressao-poly-grau{degree}")
        
        return json.dumps({
            "task": "Regressão Polinomial",
            "feature": feature,
            "target": target,
            "degree": degree,
            "metrics": {
                "MSE": round(float(mse), 4),
                "RMSE": round(float(rmse), 4),
                "R2": round(float(r2), 4)
            },
            "plot_path": plot_path
        }, indent=2)
    except Exception as e:
        logger.error(f"Erro em regression_polynomial_tool: {e}")
        return json.dumps({"error": f"Falha na regressão polinomial: {str(e)}"})


@tool
def predict_regression_tool(params: str) -> str:
    """
    Faz inferência/predição com base em um valor ou lista de valores informados pelo usuário.
    Params: target=coluna_y, feature=coluna_x, value=150.0
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        target = get_param(params_dict, "target", None)
        feature = get_param(params_dict, "feature", None)
        val_str = get_param(params_dict, "value", None)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not target and numeric_cols:
            target = numeric_cols[-1]
        if not feature and len(numeric_cols) > 1:
            feature = numeric_cols[0]
            
        if val_str is None:
            return json.dumps({"error": "Informe o parâmetro value para estimar a predição."})
            
        val = float(val_str)
        clean = df[[feature, target]].dropna()
        model = LinearRegression().fit(clean[[feature]].values, clean[target].values)
        pred = float(model.predict([[val]])[0])
        
        return json.dumps({
            "feature": feature,
            "input_value": val,
            "target": target,
            "predicted_value": round(pred, 4),
            "equation": f"{target} = {model.intercept_:.4f} + {model.coef_[0]:.4f} * {feature}"
        })
    except Exception as e:
        return json.dumps({"error": f"Erro na predição: {str(e)}"})


# ==============================================================================
# 2. FERRAMENTAS DE CLASSIFICAÇÃO (APRENDIZADO SUPERVISIONADO)
# ==============================================================================

@tool
def classification_model_tool(params: str) -> str:
    """
    Treina e avalia modelos de classificação supervisionada (KNN, Decision Tree, Random Forest ou Logistic Regression).
    Gera relatório completo de métricas (Acurácia, Precisão, Recall, F1-Score, ROC-AUC) e Matriz de Confusão gráfica.
    Params: target=coluna_classe, algorithm=knn|tree|rf|logistic, n_neighbors=5, max_depth=5, test_size=0.2
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        target = get_param(params_dict, "target", None)
        algorithm = get_param(params_dict, "algorithm", "tree").lower()
        test_size = get_param(params_dict, "test_size", 0.2, float)
        n_neighbors = get_param(params_dict, "n_neighbors", 5, int)
        max_depth = get_param(params_dict, "max_depth", 5, int)
        
        # Identificar coluna target
        if not target:
            for candidate in ["Class", "class", "target", "Target", "label", "Label", "Outcome", "y"]:
                if candidate in df.columns:
                    target = candidate
                    break
        if not target or target not in df.columns:
            return json.dumps({"error": "Coluna target de classificação não especificada ou não encontrada."})
            
        clean_df = df.dropna()
        if len(clean_df) < 10:
            clean_df = df.fillna(df.median(numeric_only=True))
            
        X = clean_df.drop(columns=[target]).select_dtypes(include=[np.number])
        if X.empty:
            return json.dumps({"error": "Não há atributos preditores numéricos para classificação."})
            
        y = clean_df[target]
        classes = [str(c) for c in np.unique(y)]
        
        # Divisão com estratificação se viável
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # Pipeline com Scaler
        if algorithm == "knn":
            clf = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))])
            alg_name = f"KNN (k={n_neighbors})"
        elif algorithm in ["tree", "decision_tree"]:
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            alg_name = f"Árvore de Decisão CART (max_depth={max_depth})"
        elif algorithm in ["rf", "random_forest"]:
            clf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
            alg_name = "Random Forest (100 árvores)"
        else:
            clf = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000))])
            alg_name = "Regressão Logística"
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        is_binary = len(classes) == 2
        avg_mode = "binary" if is_binary else "weighted"
        
        prec = precision_score(y_test, y_pred, average=avg_mode, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg_mode, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_mode, zero_division=0)
        
        roc_auc = None
        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X_test)
                if is_binary:
                    roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
            except Exception:
                roc_auc = None
                
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot da Matriz de Confusão
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Classe Prevista", fontsize=11, fontweight="bold")
        ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold")
        ax.set_title(f"Matriz de Confusão - {alg_name}\nAcurácia: {acc:.2%}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path = _save_plot(fig, prefix=f"cm-{algorithm}")
        
        # Importância de Features (se aplicável)
        feat_importances = {}
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:6]
            for idx in top_idx:
                feat_importances[X.columns[idx]] = round(float(importances[idx]), 4)
                
        return json.dumps({
            "task": "Classificação Supervisionada",
            "algorithm": alg_name,
            "target": target,
            "classes": classes,
            "test_samples": len(y_test),
            "metrics": {
                "Acuracia": f"{acc:.2%}",
                "Precisao": f"{prec:.4f}",
                "Recall": f"{rec:.4f}",
                "F1_Score": f"{f1:.4f}",
                "ROC_AUC": round(roc_auc, 4) if roc_auc is not None else "N/A"
            },
            "confusion_matrix": cm.tolist(),
            "top_features": feat_importances if feat_importances else "N/A",
            "plot_path": plot_path
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Erro em classification_model_tool: {e}")
        return json.dumps({"error": f"Falha na classificação: {str(e)}"})


@tool
def decision_tree_plot_tool(params: str) -> str:
    """
    Treina uma Árvore de Decisão e plota o diagrama da árvore (nós, regras de divisão, Gini/Entropia, amostras).
    Params: target=coluna_classe, max_depth=3, criterion=gini|entropy
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        target = get_param(params_dict, "target", None)
        max_depth = get_param(params_dict, "max_depth", 3, int)
        criterion = get_param(params_dict, "criterion", "gini")
        
        if not target:
            for candidate in ["Class", "class", "target", "Target", "label", "Label"]:
                if candidate in df.columns:
                    target = candidate
                    break
        if not target or target not in df.columns:
            return json.dumps({"error": "Target de classificação não encontrado."})
            
        clean_df = df.dropna()
        X = clean_df.drop(columns=[target]).select_dtypes(include=[np.number])
        y = clean_df[target]
        
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
        model.fit(X, y)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            model,
            feature_names=X.columns.tolist(),
            class_names=[str(c) for c in np.unique(y)],
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax
        )
        ax.set_title(f"Árvore de Decisão CART (Critério: {criterion.capitalize()}, Profundidade Máx: {max_depth})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = _save_plot(fig, prefix=f"tree-depth{max_depth}")
        
        return json.dumps({
            "message": f"Árvore de decisão gerada com critério {criterion} e max_depth={max_depth}",
            "plot_path": plot_path,
            "classes": [str(c) for c in np.unique(y)],
            "features_used": X.columns.tolist()
        })
    except Exception as e:
        logger.error(f"Erro em decision_tree_plot_tool: {e}")
        return json.dumps({"error": str(e)})


# ==============================================================================
# 3. FERRAMENTAS DE CLUSTERIZAÇÃO & NÃO SUPERVISIONADO (K-MEANS, ELBOW, SILHOUETTE, PCA)
# ==============================================================================

@tool
def cluster_evaluation_tool(params: str) -> str:
    """
    Avalia o número ótimo de clusters k (de 2 a 8) usando:
    1. Método do Cotovelo (Inércia / WCSS)
    2. Coeficiente de Silhueta (Silhouette Score)
    3. Índice Davies-Bouldin
    4. Índice Calinski-Harabasz
    Plota curvas diagnósticas completas e sugere o melhor k.
    Params: max_k=8, columns=col1|col2 (opcional)
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        max_k = get_param(params_dict, "max_k", 8, int)
        cols_param = get_param(params_dict, "columns", None)
        
        if cols_param:
            cols = [c.strip() for c in cols_param.split("|") if c.strip() in df.columns]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Se tiver muitas, limitar às primeiras 8
            cols = cols[:min(8, len(cols))]
            
        if len(cols) < 2:
            return json.dumps({"error": "Clusterização requer pelo menos 2 variáveis numéricas."})
            
        data = df[cols].dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data.values)
        
        metrics = []
        k_range = range(2, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            
            metrics.append({
                "k": k,
                "Inercia_WCSS": round(float(kmeans.inertia_), 2),
                "Silhueta": round(float(sil), 4),
                "Davies_Bouldin": round(float(db), 4),
                "Calinski_Harabasz": round(float(ch), 2)
            })
            
        metrics_df = pd.DataFrame(metrics)
        best_k_sil = int(metrics_df.loc[metrics_df["Silhueta"].idxmax()]["k"])
        
        # Plot com 2x2 subplots mostrando as 4 métricas da disciplina
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        
        # 1. Cotovelo
        axes[0, 0].plot(metrics_df["k"], metrics_df["Inercia_WCSS"], marker="o", color="blue", linewidth=2)
        axes[0, 0].set_title("1. Método do Cotovelo (Inércia / WCSS)", fontweight="bold")
        axes[0, 0].set_xlabel("Número de Clusters (k)")
        axes[0, 0].set_ylabel("Inércia (WCSS)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)
        
        # 2. Silhueta
        axes[0, 1].plot(metrics_df["k"], metrics_df["Silhueta"], marker="s", color="green", linewidth=2)
        axes[0, 1].axvline(best_k_sil, color="crimson", linestyle="--", label=f"Ótimo Silhueta: k={best_k_sil}")
        axes[0, 1].set_title("2. Coeficiente de Silhueta (Maior é Melhor)", fontweight="bold")
        axes[0, 1].set_xlabel("Número de Clusters (k)")
        axes[0, 1].set_ylabel("Score de Silhueta")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)
        axes[0, 1].legend()
        
        # 3. Davies-Bouldin
        axes[1, 0].plot(metrics_df["k"], metrics_df["Davies_Bouldin"], marker="^", color="red", linewidth=2)
        axes[1, 0].set_title("3. Índice Davies-Bouldin (Menor é Melhor)", fontweight="bold")
        axes[1, 0].set_xlabel("Número de Clusters (k)")
        axes[1, 0].set_ylabel("Índice DB")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)
        
        # 4. Calinski-Harabasz
        axes[1, 1].plot(metrics_df["k"], metrics_df["Calinski_Harabasz"], marker="D", color="purple", linewidth=2)
        axes[1, 1].set_title("4. Índice Calinski-Harabasz (Maior é Melhor)", fontweight="bold")
        axes[1, 1].set_xlabel("Número de Clusters (k)")
        axes[1, 1].set_ylabel("Score CH")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plot_path = _save_plot(fig, prefix="cluster-evaluation-metrics")
        
        return json.dumps({
            "task": "Avaliação de Clusters (Aprendizado Não Supervisionado)",
            "columns": cols,
            "metrics": metrics_df.to_dict(orient="records"),
            "recommended_k_by_silhouette": best_k_sil,
            "plot_path": plot_path
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Erro em cluster_evaluation_tool: {e}")
        return json.dumps({"error": f"Falha na avaliação de clusters: {str(e)}"})


@tool
def cluster_kmeans_plot_tool(params: str) -> str:
    """
    Executa K-Means com k definido e plota os clusters e centroides (usando 2 colunas ou PCA para visualização).
    Params: k=3, x=coluna_x, y=coluna_y
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        k = get_param(params_dict, "k", 3, int)
        x_col = get_param(params_dict, "x", None)
        y_col = get_param(params_dict, "y", None)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return json.dumps({"error": "Dataset não contém atributos numéricos."})
            
        if not x_col or x_col not in df.columns:
            x_col = numeric_cols[0]
        if not y_col or y_col not in df.columns:
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            
        data = df[[x_col, y_col]].dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data.values)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        data_plot = data.copy()
        data_plot["Cluster"] = [f"Cluster {i}" for i in labels]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=data_plot,
            x=x_col,
            y=y_col,
            hue="Cluster",
            palette="Set1",
            s=60,
            alpha=0.7,
            ax=ax
        )
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=220,
            c="black",
            marker="X",
            label="Centroides",
            edgecolor="white",
            linewidth=1.5
        )
        ax.set_title(f"Clusterização K-Means (k = {k}) com Centroides", fontsize=13, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=11, fontweight="bold")
        ax.set_ylabel(y_col, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        
        plot_path = _save_plot(fig, prefix=f"kmeans-k{k}")
        
        cluster_counts = {f"Cluster {i}": int((labels == i).sum()) for i in range(k)}
        
        return json.dumps({
            "task": "Clusterização K-Means Final",
            "k": k,
            "x_column": x_col,
            "y_column": y_col,
            "distribution": cluster_counts,
            "plot_path": plot_path
        }, indent=2)
    except Exception as e:
        logger.error(f"Erro em cluster_kmeans_plot_tool: {e}")
        return json.dumps({"error": str(e)})


@tool
def pca_reduction_tool(params: str) -> str:
    """
    Executa Redução de Dimensionalidade com PCA (Análise de Componentes Principais).
    Gera Scree Plot de variância explicada acumulada e projeção 2D (PC1 vs PC2) dos dados.
    Params: n_components=2, hue_column=Class (opcional para colorir as classes)
    """
    df = get_dataframe()
    if df is None:
        return json.dumps({"error": "Nenhum dataset carregado."})
        
    try:
        params_dict = parse_tool_params(params)
        hue_column = get_param(params_dict, "hue_column", None)
        
        # Identificar hue_column se existir
        if not hue_column:
            for c in ["Class", "class", "target", "Target", "label"]:
                if c in df.columns:
                    hue_column = c
                    break
                    
        numeric = df.select_dtypes(include=[np.number])
        if hue_column and hue_column in numeric.columns:
            numeric_features = numeric.drop(columns=[hue_column])
        else:
            numeric_features = numeric
            
        if numeric_features.shape[1] < 2:
            return json.dumps({"error": "PCA requer pelo menos 2 variáveis numéricas."})
            
        data = numeric_features.dropna()
        if len(data) > 5000:
            sample_idx = data.sample(5000, random_state=42).index
            data = data.loc[sample_idx]
        else:
            sample_idx = data.index
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data.values)
        
        # PCA completo para Scree Plot
        pca_full = PCA().fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        
        # PCA 2D
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_scaled)
        var_2d = pca_2d.explained_variance_ratio_ * 100
        
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=sample_idx)
        if hue_column and hue_column in df.columns:
            df_pca[hue_column] = df.loc[sample_idx, hue_column]
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # 1. Scree Plot
        axes[0].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="purple", linewidth=2)
        axes[0].axhline(y=80, color="crimson", linestyle="--", label="80% Variância Explicada")
        axes[0].set_title("1. Scree Plot: Variância Acumulada pelo PCA", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Número de Componentes")
        axes[0].set_ylabel("Variância Acumulada (%)")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()
        
        # 2. Scatter PC1 x PC2
        if hue_column and hue_column in df_pca.columns:
            sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue=hue_column, palette="Set1", alpha=0.7, s=50, ax=axes[1])
        else:
            axes[1].scatter(df_pca["PC1"], df_pca["PC2"], color="teal", alpha=0.6, s=40)
            
        axes[1].set_title(f"2. Projeção PCA 2D ({sum(var_2d):.1f}% variância preservada)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(f"PC1 ({var_2d[0]:.1f}% variância)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel(f"PC2 ({var_2d[1]:.1f}% variância)", fontsize=10, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plot_path = _save_plot(fig, prefix="pca-scree-projection")
        
        return json.dumps({
            "task": "PCA - Redução de Dimensionalidade",
            "original_features": numeric_features.columns.tolist(),
            "variance_explained": {
                "PC1_percent": round(float(var_2d[0]), 2),
                "PC2_percent": round(float(var_2d[1]), 2),
                "total_2D_preserved": round(float(sum(var_2d)), 2)
            },
            "plot_path": plot_path
        }, indent=2)
    except Exception as e:
        logger.error(f"Erro em pca_reduction_tool: {e}")
        return json.dumps({"error": str(e)})
