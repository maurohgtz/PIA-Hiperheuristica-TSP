"""Análisis complementario: elección del algoritmo con menor regret previsto."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .datos import ALGORITMOS, COSTOS, DISTRIBUCIONES, MEDIDAS, RESULTADOS

NUM = ["n", *MEDIDAS]
ENTRADAS = [*NUM, "distribution"]


def ajustar_distancia(train, semilla, pliegues):
    x = train[ENTRADAS]
    costos = train[COSTOS].to_numpy()
    perdidas = costos - costos.min(axis=1, keepdims=True)
    cv = StratifiedKFold(n_splits=pliegues, shuffle=True, random_state=semilla)
    partes = list(cv.split(x, train.best_algorithm))

    def crear(profundidad, hoja):
        transformar = ColumnTransformer([
            ("num", "passthrough", NUM),
            ("dist", OneHotEncoder(categories=[list(DISTRIBUCIONES)], sparse_output=False), ["distribution"]),
        ])
        arbol = DecisionTreeRegressor(criterion="squared_error", max_depth=profundidad,
                                      min_samples_leaf=hoja, random_state=semilla)
        return Pipeline([("preparar", transformar), ("regresor", arbol)])

    registros = []
    for profundidad in (1, 2, 3, 4):
        for hoja in (35, 20, 10):
            decisiones = np.empty(len(train), dtype=int)
            for ajuste, validacion in partes:
                modelo = crear(profundidad, hoja).fit(x.iloc[ajuste], perdidas[ajuste])
                decisiones[validacion] = modelo.predict(x.iloc[validacion]).argmin(axis=1)
            observado = perdidas[np.arange(len(train)), decisiones]
            etiquetas = np.asarray(ALGORITMOS)[decisiones]
            registros.append({"max_depth": profundidad, "min_samples_leaf": hoja,
                              "regret_cv": float(observado.mean()),
                              "aciertos_cv": int((etiquetas == train.best_algorithm).sum())})
    tabla = pd.DataFrame(registros).sort_values(
        ["regret_cv", "aciertos_cv", "max_depth", "min_samples_leaf"], ascending=[True, False, True, False]
    )
    tabla.to_csv(RESULTADOS / "validacion_distancia.csv", index=False)
    config = {c: int(tabla.iloc[0][c]) for c in ("max_depth", "min_samples_leaf")}
    modelo = crear(config["max_depth"], config["min_samples_leaf"]).fit(x, perdidas)
    arbol = modelo.named_steps["regresor"].tree_
    selector = {
        "configuracion": {"familia": "CART", "objetivo": "distancia", **config},
        "numericas": NUM, "categorias": list(DISTRIBUCIONES), "clases": list(ALGORITMOS),
        "izq": arbol.children_left.tolist(), "der": arbol.children_right.tolist(),
        "atributo": arbol.feature.tolist(), "umbral": arbol.threshold.tolist(),
        "clase_nodo": np.argmin(arbol.value[:, :, 0], axis=1).tolist(),
    }
    return modelo, selector
