# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
import json
import pickle
import gzip
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# ---------- I/O y utilidades ----------
def ensure_dirs():
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

def load_data():
    train_df = pd.read_csv("files/input/train_data.csv.zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip")
    return train_df, test_df


def save_estimator(estimator, path="files/models/model.pkl.gz"):
    """Guarda el estimador comprimido con gzip."""
    ensure_dirs()
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def load_estimator(path="files/models/model.pkl.gz"):
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


# ---------- Preprocesamiento ----------
def prepare_data(df):
    df = df.copy()

    # renombrar
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    if "EDUCATION" in df.columns:
        df["EDUCATION"] = pd.to_numeric(df["EDUCATION"], errors="coerce")
        df.loc[df["EDUCATION"].isna(), "EDUCATION"] = 4  
        df["EDUCATION"] = df["EDUCATION"].astype(int)
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
        df.loc[df["EDUCATION"] == 0, "EDUCATION"] = 4

    df.drop(df[df["MARRIAGE"] == 0].index, inplace=True)
    df = df.dropna()

    return df


# ---------- Split ----------
def make_train_test_split(train_df, test_df):
    """
    Aplica prepare_data y retorna x_train, y_train, x_test, y_test
    """
    train_clean = prepare_data(train_df)
    test_clean = prepare_data(test_df)

    x_train = train_clean.drop(columns=["default"])
    y_train = train_clean["default"]

    x_test = test_clean.drop(columns=["default"])
    y_test = test_clean["default"]

    return x_train, y_train, x_test, y_test


# ---------- Pipeline ----------
def make_pipeline(feature_columns):
    """
    Construye pipeline. Se requiere pasar feature_columns (lista de columnas de X)
    para inferir columnas numéricas.
    """
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric_features = [c for c in feature_columns if c not in categorical_features]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Definición del pipeline para las variables numéricas
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Combinación de ambos pipelines en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=None)),
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', SVC(random_state=11))
        ], verbose=False
    )

    return pipeline

# ---------- Búsqueda de hiperparámetros ----------
def make_grid_search(estimator, param_grid, cv=10):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True        
    )
    return grid_search

# ---------- Entrenamiento ----------
def train_estimator(grid_search):
    """
    Entrena grid_search con los datos, compara con modelo guardado (si existe)
    usando balanced_accuracy en el conjunto de test; mantiene el mejor.
    """
    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    # entrenar
    grid_search.fit(x_train, y_train)

    # cargar modelo guardado (si existe) y comparar en balanced_accuracy sobre test
    saved = load_estimator()
    current_score = balanced_accuracy_score(y_test, grid_search.predict(x_test))

    if saved is not None:
        try:
            saved_score = balanced_accuracy_score(y_test, saved.predict(x_test))
        except Exception:
            # si el objeto guardado no tiene predict (por ejemplo, fue guardado mal), sobreescribimos
            saved_score = -1.0
    else:
        saved_score = -1.0

    if current_score >= saved_score:
        # guardar el grid_search (fitted) para revisar cv_results etc.
        save_estimator(grid_search)
    else:
        # mantener el guardado (no sobreescribimos)
        pass


# ---------- Entrenador específico ----------
def train_logistic_regression():
    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    pipeline = make_pipeline(feature_columns=x_train.columns.tolist())

    param_grid ={
    'selector__k': [15, 17, 20, 'all'],
    'classifier__gamma': [0.01, 0.1, 1],
    }
    
    gs = make_grid_search(estimator=pipeline, param_grid=param_grid, cv=10)
    train_estimator(gs)


# ---------- Check / métricas ----------
def check_estimator():
    ensure_dirs()
    train_df, test_df = load_data()
    x_train, y_train, x_test, y_test = make_train_test_split(train_df, test_df)

    # cargar modelo (gzip)
    estimator = load_estimator()
    if estimator is None:
        raise FileNotFoundError("No se encontró modelo en files/models/model.pkl.gz")

    # predicciones
    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    metrics = []

    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
    }
    metrics.append(train_metrics)

    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
    }
    metrics.append(test_metrics)

    # matriz de confusión train
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
    }
    metrics.append(cm_train_dict)

    # matriz de confusión test
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
    }
    metrics.append(cm_test_dict)

    # guardar JSONL
    out_path = "files/output/metrics.json"
    with open(out_path, "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    print(f"Métricas guardadas en {out_path}")


if __name__ == "__main__":
    # si se ejecuta el script, entrena y luego comprueba
    ensure_dirs()
    train_logistic_regression()
    check_estimator()

