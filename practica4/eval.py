import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import multilabel_confusion_matrix



# CWD = os.getcwd()
CWD = os.path.dirname(__file__)
TEST_DATA_PATH = os.path.join(CWD, "dataset", "test")

# Ruta donde se guardaron los entrenamientos del paso anterior
RUNS_PATH = os.path.join(CWD, "runs")

# nombres de las clases ordenados alfabeticamente
CLASS_NAMES = sorted(os.listdir(TEST_DATA_PATH))
print(CLASS_NAMES)


# Ruta donde guardar matrices de confusión
TEST_RESULTS_PATH = os.path.join(CWD, "test_results")

if not os.path.exists(TEST_RESULTS_PATH):
    print(f"Creando directorio {TEST_RESULTS_PATH}.")
    os.mkdir(TEST_RESULTS_PATH)

# Nombres de las métricas que vamos a obtener de cada modelo
METRICS_NAMES = ["acc", "recall", "precision", "FNR", "FPR", "specificity", "F1", "AUC"]



def make_predictions(model_path, test_path, class_list):
    """Compute predictions with the trained model and return y_true, y_pred, y_score"""

    model = YOLO(model_path)
    # print(model.names)

    y_true = []
    y_pred = []
    y_score = []
    
    # Iteramos sobre cada carpeta de clase (MEL, NV, etc.)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_path, class_name)
        if not os.path.isdir(class_dir): continue
            
        # Obtenemos el índice numérico de la clase (ej: MEL = 0)
        if class_name in class_list:
            class_idx = class_list.index(class_name)
        else:
            continue
            
        # Listamos imágenes
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

        # Predecimos
        results = model.predict(images, verbose=False, imgsz=224)

        for res in results:
            # probs.top1 devuelve el índice de la clase con mayor probabilidad
            pred_idx = res.probs.top1
            y_score.append(res.probs.data.cpu().numpy())
            y_true.append(class_idx)
            y_pred.append(pred_idx)

    return np.array(y_true), np.array(y_pred), np.array(y_score)



def get_confusion_matrix(y_true, y_pred, model_name):
    """Function to generate the confusion matrix of the model given its predictions"""
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    # La mostramos en el terminal
    print("Matriz de confusión:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=CLASS_NAMES
    )
    disp.plot()

    # Guardamos la matriz de confunsión en una imagen (p.ej "CM_yolo11n.png")
    plt.savefig(os.path.join(TEST_RESULTS_PATH, f"CM_{model_name}.png"))



def compute_metrics(y_true, y_pred):
    """Function to compute all the necessary metrics using the "One vs Rest" approach"""
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    print(mcm)
    
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]

    accuracy = accuracy_score(y_true, y_pred)
    recall = np.mean(tp / (tp + fn))
    precision = np.mean(tp / (tp + fp))
    fnr = np.mean(fn / (fn + tp))
    fpr = np.mean(fp / (fp + tn))
    specificity = 1 - fpr
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, recall, precision, fnr, fpr, specificity, f1



def get_ROC_curve_and_AUC(y_true, y_score, model_name):
    """
    Function to generate the ROC curve of the model using the "One vs Rest" approach and
    return the mean of the AUC values for each positive class
    """

    fig, ax = plt.subplots()

    for class_id in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true, y_score[:, class_id], pos_label=class_id)  
        ax.plot(
            fpr, tpr, 
            label=f"ROC curve - Positive class: {CLASS_NAMES[class_id]}"
        )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Curva ROC One-vs-Rest: {model_name}"
    )
    
    ax.legend()
    fig.savefig(os.path.join(TEST_RESULTS_PATH, f"ROC_{model_name}.png")) 

    return np.mean(roc_auc_score(y_true, y_score, average=None, multi_class="ovr"))   



def main():

    metrics = {}
    models = ["yolo11n", "yolo11s", "yolo11m"]

    for variant in models:
        # Construimos path al archivo best.pt
        model_path = os.path.join(RUNS_PATH, f'train_{variant}-cls.pt', 'weights', 'best.pt')
        
        if not os.path.exists(model_path):
            print(f"No se encontró el modelo {variant} en {model_path}")
            continue
            
        print(f"\n-------------------Evaluando {variant}-------------------")
        print(f"Ruta del modelo: {model_path}")

        # Obtener las predicciones para el conjunto de test
        y_true, y_pred, y_score = make_predictions(model_path, TEST_DATA_PATH, CLASS_NAMES)

        # print(y_true)
        # print(y_pred)


        # Matriz de confusión del modelo
        get_confusion_matrix(y_true, y_pred, model_name=variant)
        
        # Calculamos todas las métricas de rendimiento necesarias
        acc, recall, precision, fnr, fpr, specificity, f1 = \
            compute_metrics(y_true=y_true, y_pred=y_pred)
        
        # Obtenemos la curva ROC
        auc = get_ROC_curve_and_AUC(y_true, y_score, model_name=variant)

        # Guardamos las métricas
        metrics[variant] = [acc, recall, precision, fnr, fpr, specificity, f1, auc]

        
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