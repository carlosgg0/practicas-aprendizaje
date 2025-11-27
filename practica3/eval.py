# En este archivo se implementa un método que, dado un modelo entrenado y un conjunto de test, 
# realiza las predicciones del modelo, y calcula en rendimiento del mismo

import pandas as pd
import numpy as np

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