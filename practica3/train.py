# En este archivo se implementa el método que devuelve un modelo entrenado
# para la partición de la versión del conjunto de datos que se introduzca como parámetro

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def cargar_datos_entrenamiento(set, fold):
    """
        Carga el conjunto de datos de entrenamiento en un dataFrame.
    
        Args:
            set: version del conjunto de datos
            fold: particion
                    
        Returns: 
            training_data: el dataFrame con el conjunto de entrenamiento
    """
    training_data = pd.read_csv(f"./{set}/training{fold}_{set}.csv")
    return training_data


def entrena_modelo(model_type, set, fold):
    """
        Entrena un modelo seleccionado, para la version del conjunto y la partición seleccionadas.
    
        Args:
            model: tipo de modelo
            set: version del conjunto de datos
            fold: particion
                    
        Returns: 
            model_fitted: modelo entrenado con los conjuntos de datos
    """

    training_data = cargar_datos_entrenamiento(set, fold)
    X_train = training_data.iloc[:,:-1]
    y_train = training_data.iloc[:,-1]

    if model_type == 'knn':                 # K-Nearest Neighbors (KNN)
        model_fitted = KNeighborsClassifier(n_neighbors=5)

    elif model_type == 'svm':               # Support Vector Machine (SVM)
        model_fitted = SVC(kernel='linear', random_state=42)

    elif model_type == 'naive_bayes':       # Naive Bayes (Gaussian)
        model_fitted = GaussianNB()

    elif model_type == 'random_forest':     # Random Forest
        model_fitted = RandomForestClassifier(n_estimators=100, random_state=42)

    model_fitted.fit(X_train, y_train)
    
    return model_fitted
