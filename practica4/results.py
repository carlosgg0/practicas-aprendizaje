import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import multilabel_confusion_matrix





def main():
    models = ["yolo11n", "yolo11s", "yolo11m"]
    metrics = {}
    for variant in models:

        y_true = pd.read_csv()
        
        # Matriz de confusión del modelo
        # get_confusion_matrix(y_true, y_pred, model_name=variant)
        
        # Calculamos todas las métricas de rendimiento necesarias
        acc, recall, precision, fnr, fpr, specificity, f1 = \
            compute_metrics(y_true=y_true, y_pred=y_pred)
        
        # Obtenemos la curva ROC
        # auc = get_ROC_curve_and_AUC(y_true, y_score, model_name=variant)

        # Guardamos las métricas
        metrics[variant] = [acc, recall, precision, fnr, fpr, specificity, f1]


    # Creamos un dataframe con todas las métricas
    df_metrics = pd.DataFrame(
        metrics,
        index=METRICS_NAMES
    ).T

    print(df_metrics)

    # Convertimos el dataframe a latex y a csv
    df_metrics.to_latex(
        os.path.join(TEST_RESULTS_PATH, "metrics.tex"),
        caption="Rendimiento de las tres versiones de YOLO"
    )

    df_metrics.to_csv(os.path.join(TEST_RESULTS_PATH, "metrics.csv"))

if __name__ == "__main__":
    main()