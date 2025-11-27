import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

RANDOM = 42


# ESTE SCRIPT GENERA LOS DATASETS CON SHUFFLE


def generate_train_test_set(kf, data, target, name):
    
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "data")
    
    try:
        os.mkdir(path=os.path.join(data_path, name))
    except FileExistsError:
        pass

    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        
        X_train = data[train_idx]
        X_test = data[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]

        training = pd.DataFrame(
            np.concatenate((X_train, y_train.reshape(y_train.shape[0], 1)), axis=1)
        )
        test = pd.DataFrame(
            np.concatenate((X_test, y_test.reshape(y_test.shape[0], 1)), axis=1)
        )

        training_path = os.path.join(data_path, f"{name}/training{fold + 1}_{name}.csv")
        print("Training path:", training_path)

        test_path = os.path.join(data_path, f"{name}/test{fold + 1}_{name}.csv")
        print("Test path:", test_path)
        
        training.to_csv(training_path, index=False)
        test.to_csv(test_path, index=False)



def main():

    iris = datasets.load_iris()
    original = iris.data   
    y = iris.target


    original, y = shuffle(original, y, random_state=RANDOM)

    standarized = StandardScaler().fit_transform(original)
    normalized = MinMaxScaler().fit_transform(original)

    pca95 = PCA(n_components=0.95)
    pca80 = PCA(n_components=0.8)

    original_pca95 = pca95.fit_transform(original)
    standarized_pca95 = pca95.fit_transform(standarized)
    normalized_pca95 = pca95.fit_transform(normalized)

    original_pca80 = pca80.fit_transform(original)
    standarized_pca80 = pca80.fit_transform(standarized)
    normalized_pca80 = pca80.fit_transform(normalized)

    n_splits = 5
    kf = KFold(n_splits=n_splits)

    # Original, Normalizado y Estandarizado
    generate_train_test_set(kf, data=original, target=y, name="original")
    generate_train_test_set(kf, data=normalized, target=y, name="norm")
    generate_train_test_set(kf, data=standarized, target=y, name="stand")

    # Original PCA
    generate_train_test_set(kf, data=original_pca80, target=y, name="original_PCA80")
    generate_train_test_set(kf, data=original_pca95, target=y, name="original_PCA95")

    # Normalizado PCA
    generate_train_test_set(kf, data=normalized_pca80, target=y, name="norm_PCA80")
    generate_train_test_set(kf, data=normalized_pca95, target=y, name="norm_PCA95")

    # Estandarizado PCA
    generate_train_test_set(kf, data=standarized_pca80, target=y, name="stand_PCA80")
    generate_train_test_set(kf, data=standarized_pca95, target=y, name="stand_PCA95")

if __name__ == "__main__":
    main()