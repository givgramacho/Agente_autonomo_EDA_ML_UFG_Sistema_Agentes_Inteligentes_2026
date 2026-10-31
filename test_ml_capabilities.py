#!/usr/bin/env python3
"""
test_ml_capabilities.py
Validação automatizada de todas as capacidades de Machine Learning implementadas:
1. Regressão Linear & Polinomial (MSE, RMSE, R², Equação da Reta)
2. Classificação Supervisionada (KNN, Decision Tree, Acurácia, Matriz de Confusão)
3. Clusterização Não Supervisionada (K-Means, Cotovelo, Silhueta, PCA)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Garante suporte a utf-8 no console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tools import set_dataframe
from tools_ml import (
    regression_linear_tool,
    regression_polynomial_tool,
    predict_regression_tool,
    classification_model_tool,
    decision_tree_plot_tool,
    cluster_evaluation_tool,
    cluster_kmeans_plot_tool,
    pca_reduction_tool,
)


def run_tests():
    print("=" * 70)
    print("[TESTES] INICIANDO TESTES DO PACOTE DE MACHINE LEARNING DOS AGENTES")
    print("=" * 70)

    # 1. Criar dataset sintético para teste com regressão, classificação e agrupamento
    np.random.seed(42)
    n = 150
    area = np.random.uniform(40, 300, n)
    quartos = np.random.randint(1, 5, n)
    # Preço com relação linear + ruído (para regressão)
    preco = 50 + 2.5 * area + 30 * quartos + np.random.normal(0, 15, n)
    # Classe binária (ex: Alto padrão vs Normal)
    classe = (preco > np.median(preco)).astype(int)

    df_test = pd.DataFrame({
        "Area_m2": area,
        "Quartos": quartos,
        "Preco_Milhares": preco,
        "Class": classe
    })

    set_dataframe(df_test)
    print(f"[OK] Dataset de teste carregado: {df_test.shape[0]} amostras, colunas: {list(df_test.columns)}")

    # -------------------------------------------------------------
    # Teste 1: Regressão Linear
    # -------------------------------------------------------------
    print("\n[1/8] Testando regression_linear_tool...")
    res_reg = regression_linear_tool("target=Preco_Milhares, features=Area_m2|Quartos")
    data_reg = json.loads(res_reg)
    assert "metrics" in data_reg, f"Falha na regressão: {res_reg}"
    print(f"   * R2: {data_reg['metrics']['R2']}, MSE: {data_reg['metrics']['MSE']}")
    print(f"   * Equacao: {data_reg['equation']}")
    print(f"   * Grafico: {data_reg.get('plot_path')}")

    # -------------------------------------------------------------
    # Teste 2: Regressão Polinomial
    # -------------------------------------------------------------
    print("\n[2/8] Testando regression_polynomial_tool...")
    res_poly = regression_polynomial_tool("target=Preco_Milhares, feature=Area_m2, degree=2")
    data_poly = json.loads(res_poly)
    assert "metrics" in data_poly, f"Falha na regressão polinomial: {res_poly}"
    print(f"   * Polinomio Grau 2 R2: {data_poly['metrics']['R2']}")
    print(f"   * Grafico: {data_poly.get('plot_path')}")

    # -------------------------------------------------------------
    # Teste 3: Inferência de Regressão
    # -------------------------------------------------------------
    print("\n[3/8] Testando predict_regression_tool...")
    res_pred = predict_regression_tool("target=Preco_Milhares, feature=Area_m2, value=120.0")
    data_pred = json.loads(res_pred)
    assert "predicted_value" in data_pred
    print(f"   * Area: 120m2 -> Preco Previsto: {data_pred['predicted_value']}")

    # -------------------------------------------------------------
    # Teste 4: Classificação com Árvore de Decisão CART
    # -------------------------------------------------------------
    print("\n[4/8] Testando classification_model_tool (Decision Tree)...")
    res_clf_tree = classification_model_tool("target=Class, algorithm=tree, max_depth=4")
    data_clf_tree = json.loads(res_clf_tree)
    assert "metrics" in data_clf_tree, f"Falha na classificação: {res_clf_tree}"
    print(f"   * Acuracia: {data_clf_tree['metrics']['Acuracia']}, F1: {data_clf_tree['metrics']['F1_Score']}")
    print(f"   * Matriz de Confusao: {data_clf_tree['confusion_matrix']}")

    # -------------------------------------------------------------
    # Teste 5: Classificação com KNN
    # -------------------------------------------------------------
    print("\n[5/8] Testando classification_model_tool (KNN)...")
    res_clf_knn = classification_model_tool("target=Class, algorithm=knn, n_neighbors=5")
    data_clf_knn = json.loads(res_clf_knn)
    assert "metrics" in data_clf_knn
    print(f"   * KNN Acuracia: {data_clf_knn['metrics']['Acuracia']}")

    # -------------------------------------------------------------
    # Teste 6: Plot da Árvore de Decisão
    # -------------------------------------------------------------
    print("\n[6/8] Testando decision_tree_plot_tool...")
    res_tree_plot = decision_tree_plot_tool("target=Class, max_depth=3")
    data_tree_plot = json.loads(res_tree_plot)
    assert "plot_path" in data_tree_plot
    print(f"   * Diagrama da Arvore: {data_tree_plot['plot_path']}")

    # -------------------------------------------------------------
    # Teste 7: Avaliação de Clusters (Cotovelo, Silhueta, DB, CH)
    # -------------------------------------------------------------
    print("\n[7/8] Testando cluster_evaluation_tool...")
    res_eval = cluster_evaluation_tool("max_k=6")
    data_eval = json.loads(res_eval)
    assert "metrics" in data_eval
    print(f"   * k recomendado pela Silhueta: {data_eval['recommended_k_by_silhouette']}")
    print(f"   * Grafico 4 metricas: {data_eval.get('plot_path')}")

    # -------------------------------------------------------------
    # Teste 8: K-Means Plot e PCA
    # -------------------------------------------------------------
    print("\n[8/8] Testando cluster_kmeans_plot_tool e pca_reduction_tool...")
    res_kmeans = cluster_kmeans_plot_tool("k=3, x=Area_m2, y=Preco_Milhares")
    data_kmeans = json.loads(res_kmeans)
    print(f"   * K-Means Clusters: {data_kmeans['distribution']}")

    res_pca = pca_reduction_tool("hue_column=Class")
    data_pca = json.loads(res_pca)
    print(f"   * PCA 2D Variancia Preservada: {data_pca['variance_explained']['total_2D_preserved']}%")

    print("\n" + "=" * 70)
    print("[SUCESSO] TODOS OS 8 TESTES DE MACHINE LEARNING PASSARAM COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
