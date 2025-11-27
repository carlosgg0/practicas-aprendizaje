# En este archivo se implementa un método que, dado un modelo entrenado y un conjunto de test, 
# realiza las predicciones del modelo, y calcula en rendimiento del mismo

import pandas as pd
import numpy as np
import os
import train
from sklearn.metrics import roc_curve, roc_auc_score

MODELS = ['knn', 'svm', 'naive_bayes', 'random_forest']
SETS = ['norm', 'norm_PCA80', 'norm_PCA95', 'original', 'original_PCA80', 'original_PCA95', 'stand', 'stand_PCA80', 'stand_PCA95']
METRICS_NAMES = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'FNR', 'FPR', 'AUC']

def cargar_datos_prueba(set, fold):
    """
        Carga el conjunto de datos de prueba en un dataFrame.

        Args:
            set: version del conjunto de datos
            fold: particion

        Returns: 
            test_data: el dataFrame con el conjunto de prueba
    """
    test_data = pd.read_csv(f"./{set}/test{fold}_{set}.csv")
    return test_data

def evaluar_modelo(model, set, fold):
    """
        Evalúa el modelo en cuestión utilizando Cross Validation con K iteraciones.

        Args:
            model: nombre del método a utilizar
            k: número de iteraciones

        Returns:
            metrics: una lista de las 8 métricas obtenidas
    """
    # Cargamos el conjunto de validación
    test_data = pd.read_csv(f"./{set}/test{fold}_{set}.csv")

    X_test = test_data.iloc[:,:-1]
    y_test = test_data.iloc[:,-1]

    # Obtenemos los valores predichos por el modelo
    y_pred = model.predict(X_test)

    # Obtenemos las probabilidades de pertenencia a cada clase
    y_scores = model.predict_proba(X_test)

    # Cálculo de AUC_ROC
    roc_auc = roc_auc_score(y_test, y_scores)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_scores)

    # Calculamos los valores de la matriz de confusión
    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    # Calculamos métricas
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    fnr = FN / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return [accuracy, sensitivity, specificity, precision, f1, fnr, fpr, roc_auc]

def obtener_metricas():
    # Iteramos sobre todos los modelos y sets
    for model in MODELS:
        for version in SETS:
            fold_metrics = []
            for fold in range(1,6):
                eval_model = train.entrena_modelo(model, version, fold)
                metrics = evaluar_modelo(eval_model, version, fold)

                fold_metrics.append(metrics)
                
            # Creamos el DataFrame con los resultados de los folds
            df_metrics = pd.DataFrame(fold_metrics, columns=METRICS_NAMES)
            
            # Calculamos la media de cada columna (métrica)
            df_mean = df_metrics.mean().to_frame().T
            df_mean.index = ['Mean']
            
            # Concatenamos la media al final del DataFrame
            df_final = pd.concat([df_metrics, df_mean])
            
            # Asignamos nombres (Folds 1-5 y Mean)
            df_final.index = [f'Fold_{i}' for i in range(1, 6)] + ['Mean']

            # Exportamos el CSV (creamos la carpeta si no existe)
            OUTPUT_DIR = 'metrics'
    
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
                print(f"Carpeta '{OUTPUT_DIR}' creada.")

            file_name = f"{model}_{version}_metrics.csv"
            full_path = os.path.join(OUTPUT_DIR, file_name)
            df_final.to_csv(full_path, index=True, index_label='Fold/Mean')
            
            print(f"Guardado: {file_name}")

if __name__ == "__main__":
    obtener_metricas()