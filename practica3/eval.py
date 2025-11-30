# En este archivo se implementa un método que, dado un modelo entrenado y un conjunto de test, 
# realiza las predicciones del modelo, y calcula en rendimiento del mismo

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix


TARGET_NAMES = ["setosa", "versicolor", "virginica"]        # Contiene los nombres de las clases
METHODS = ['knn', 'svm', 'naive_bayes', 'random_forest']    # Nombre de todos los métodos 
SETS = [                                                    # Nombre de todos los conjuntos de datos
    'norm', 'norm_PCA80', 'norm_PCA95',
    'original', 'original_PCA80', 'original_PCA95',
    'stand', 'stand_PCA80', 'stand_PCA95'
]
# Nombre de todas las métricas a calcular
METRICS_NAMES = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'FNR', 'FPR', 'AUC']

# Creación de directorios necesarios
CWD = os.getcwd()
PREDICTIONS_PATH = os.path.join(CWD, "predictions")
IMAGES_PATH = os.path.join(CWD, "roc-curve-images")
DATA_PATH = os.path.join(CWD, "data")
METRICS_PATH = os.path.join(CWD, "metrics")

MODELS_PATH = os.path.join(CWD, "models") # Este se creó en train.ipynb

try:
    os.mkdir(PREDICTIONS_PATH)
    os.mkdir(IMAGES_PATH)
    os.mkdir(DATA_PATH)
    os.mkdir(METRICS_PATH)
except FileExistsError:
    pass

def make_predictions(
    model: KNeighborsClassifier | SVC | GaussianNB | RandomForestClassifier,
    set: str, 
    fold: int,
    method_name: str
):
    """
    Esta función se encarga de:
    - Hacer las predicciones de un modelo sobre un conjunto de test y fold determinado
    - Guardar las probabilidades de pertenencia a cada clase en formato csv
    - Generar curvas ROC y el área bajo la curva
    - Generar métricas accuracy, precision, etc. y las devuelve como una lista
    Nótese que las métricas son calculadas "One-vs-rest", es decir, 
    se calculan todas las métricas como si fuera un problema binario
    considerando una de las clases como positiva y las otras dos negativas
    y luego se computa la media de las métricas obtenidas
    """
    print(f"model: {method_name}, set: {set}, fold: {fold}")
    
    # Cargamos el conjunto de datos de test del fold correspondiente
    full_path = os.path.join(DATA_PATH, set, f"test{fold}_{set}.csv")
    test_df = pd.read_csv(full_path)
    
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    y_pred = model.predict(X_test)
    
    # Probabilidades de pertenencia a cada clase
    y_scores = pd.DataFrame(model.predict_proba(X_test))
    y_scores.to_csv(os.path.join(PREDICTIONS_PATH, f"pred_{fold}_{set}_{method_name}.csv"), index=False)
    
    # Curvas ROC del random forest:
    if method_name == "random_forest":
        fig, ax = plt.subplots(figsize=(6, 6))

        for class_id in range(3):
            fpr, tpr, _ = roc_curve(y_test, y_scores.iloc[ : , class_id], pos_label=class_id)  
            ax.plot(fpr, tpr, label=f"ROC curve - Positive class: {TARGET_NAMES[class_id]}")

        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Curva ROC One-vs-Rest"
        )
        
        ax.legend()
        fig.savefig(os.path.join(IMAGES_PATH, f"ROC_{set}_{fold}.png"))
        plt.close(fig) # No mostrar la figura

    # Área debajo de la curva ROC
    roc_auc = roc_auc_score(y_test, y_scores, multi_class='ovr')
    
    # Resto de métricas
    cm = multilabel_confusion_matrix(y_test, y_pred)
    tn = cm[ : , 0, 0]
    tp = cm[ : , 1, 1]
    fn = cm[ : , 1, 0]
    fp = cm[ : , 0, 1]

    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = np.mean(tp / (tp + fn))
    recall = sensitivity
    precision = np.mean(tp / (tp + fp))
    fnr = np.mean(fn / (fn + tp))
    fpr = np.mean(fp / (fp + tn))
    specificity = 1 - fpr
    f1 = 2 * precision * recall / (precision + recall)
    
    return [accuracy, sensitivity, specificity, precision, f1, fnr, fpr, roc_auc]