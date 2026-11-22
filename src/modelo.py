"""Selección del clasificador con la partición train y evaluación en test."""
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .distancia import ENTRADAS, ajustar_distancia
from .datos import ALGORITMOS, COSTOS, DISTRIBUCIONES, EXTRAS, MEDIDAS, RESULTADOS

SEMILLA = 20261122
PLIEGUES = 5
PERMUTACIONES = 40


def guardar_json(nombre, valor):
    (RESULTADOS / nombre).write_text(
        json.dumps(valor, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def numericas(config):
    return ["n", *MEDIDAS, *(EXTRAS if config["rasgos"] == "ampliados" else ())]


def columnas(config):
    return [*numericas(config), "distribution"]


def candidatos():
    for rasgos in ("curso", "ampliados"):
        for profundidad, hoja in product((1, 2, 3, 5), (15, 5)):
            yield {"familia": "CART", "rasgos": rasgos, "max_depth": profundidad, "min_samples_leaf": hoja}
        for k in (7, 15, 31):
            yield {"familia": "KNN", "rasgos": rasgos, "k": k}


def construir(config):
    if config["familia"] == "CART":
        escala = "passthrough"
        clasificador = DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"], random_state=SEMILLA
        )
    else:
        escala = StandardScaler()
        clasificador = KNeighborsClassifier(
            n_neighbors=config["k"], metric="euclidean", weights="uniform", algorithm="brute"
        )
    preparar = ColumnTransformer([
        ("num", escala, numericas(config)),
        ("dist", OneHotEncoder(categories=[list(DISTRIBUCIONES)], sparse_output=False), ["distribution"])
    ])
    return Pipeline([("preparar", preparar), ("clasificar", clasificador)])


def costos_elegidos(tabla, elecciones):
    posiciones = np.array([ALGORITMOS.index(a) for a in elecciones])
    return tabla[COSTOS].to_numpy(dtype=float)[np.arange(len(tabla)), posiciones]


def regret(tabla, elecciones):
    diferencia = costos_elegidos(tabla, elecciones) - tabla[COSTOS].min(axis=1).to_numpy()
    return np.round(diferencia, 12)


def seleccionar(train):
    y = train["best_algorithm"].to_numpy()
    cv = StratifiedKFold(n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA)
    particiones = list(cv.split(train, y))
    registros = []
    configs = list(candidatos())
    for orden, config in enumerate(configs):
        x = train[columnas(config)]
        pred = np.empty(len(train), dtype=object)
        for indices_fit, indices_val in particiones:
            modelo = construir(config)
            modelo.fit(x.iloc[indices_fit], y[indices_fit])
            pred[indices_val] = modelo.predict(x.iloc[indices_val])
        registros.append({"orden": orden, **config, "aciertos_cv": int((pred == y).sum()),
                          "accuracy_cv": float((pred == y).mean()), "regret_cv": float(regret(train, pred).mean())})
    tabla = pd.DataFrame(registros).sort_values(
        ["aciertos_cv", "regret_cv", "orden"], ascending=[False, True, True], kind="stable"
    )
    tabla.to_csv(RESULTADOS / "validacion_cruzada.csv", index=False)
    config = configs[int(tabla.iloc[0]["orden"])]
    final = construir(config).fit(train[columnas(config)], y)
    return config, final


def exportar(modelo, config, train):
    """Formato portátil para aplicar el selector con Python y NumPy."""
    clasificador = modelo.named_steps["clasificar"]
    preparar = modelo.named_steps["preparar"]
    salida = {"configuracion": config, "numericas": numericas(config),
              "categorias": list(DISTRIBUCIONES), "clases": clasificador.classes_.tolist()}
    if config["familia"] == "CART":
        arbol = clasificador.tree_
        salida.update(izq=arbol.children_left.tolist(), der=arbol.children_right.tolist(),
                      atributo=arbol.feature.tolist(), umbral=arbol.threshold.tolist(),
                      clase_nodo=np.argmax(arbol.value[:, 0, :], axis=1).tolist())
    else:
        escala = preparar.named_transformers_["num"]
        salida.update(media=escala.mean_.tolist(), escala=escala.scale_.tolist(),
                      ejemplos=preparar.transform(train[columnas(config)]).tolist(),
                      etiquetas=train["best_algorithm"].tolist())
    guardar_json("selector.json", salida)
    return salida


def predecir_fila(selector, fila):
    """Aplica la configuración ganadora a una instancia individual."""
    valores = np.asarray([fila[c] for c in selector["numericas"]], dtype=float)
    dist = fila["distribution"]
    if dist not in selector["categorias"] or not np.isfinite(valores).all():
        raise ValueError("Características incompletas o distribución desconocida.")
    indicadores = [float(dist == cat) for cat in selector["categorias"]]
    if selector["configuracion"]["familia"] == "CART":
        x = np.concatenate((valores, indicadores)).astype(np.float32)
        nodo = 0
        while selector["izq"][nodo] != -1:
            columna = selector["atributo"][nodo]
            nodo = (selector["izq"][nodo] if x[columna] <= selector["umbral"][nodo] else selector["der"][nodo])
        return selector["clases"][selector["clase_nodo"][nodo]]
    z = (valores - np.asarray(selector["media"])) / np.asarray(selector["escala"])
    x = np.concatenate((z, indicadores))
    distancias = np.sum((np.asarray(selector["ejemplos"]) - x) ** 2, axis=1)
    vecinos = np.argsort(distancias, kind="stable")[:selector["configuracion"]["k"]]
    votos = [selector["etiquetas"][i] for i in vecinos]
    return max(selector["clases"], key=votos.count)


def importancia(modelo, config, test, accuracy):
    """Cada permutación rompe la asociación de una variable completa con la instancia."""
    x = test[columnas(config)].copy()
    y = test["best_algorithm"].to_numpy()
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    for atributo in x:
        perdidas = []
        for _ in range(PERMUTACIONES):
            permutado = x.copy()
            permutado[atributo] = rng.permutation(x[atributo].to_numpy())
            acierto = (modelo.predict(permutado) == y).mean()
            perdidas.append(100 * (accuracy - acierto))
        filas.append({"atributo": atributo, "caida_pp": np.mean(perdidas), "desv_pp": np.std(perdidas, ddof=1)})
    pd.DataFrame(filas).sort_values("caida_pp", ascending=False).to_csv(RESULTADOS / "importancias.csv", index=False)


def importancia_distancia(modelo, test, regret_observado):
    x = test[ENTRADAS]
    rng = np.random.default_rng(SEMILLA + 1)
    filas = []
    for atributo in x:
        aumentos = []
        for _ in range(PERMUTACIONES):
            modificado = x.copy()
            modificado[atributo] = rng.permutation(x[atributo].to_numpy())
            decisiones = np.asarray(ALGORITMOS)[modelo.predict(modificado).argmin(axis=1)]
            aumentos.append(regret(test, decisiones).mean() - regret_observado)
        filas.append({"atributo": atributo, "aumento_regret": np.mean(aumentos),
                      "desv_regret": np.std(aumentos, ddof=1)})
    pd.DataFrame(filas).sort_values("aumento_regret", ascending=False).to_csv(
        RESULTADOS / "importancias_distancia.csv", index=False
    )


def matriz(real, pred):
    conteos = confusion_matrix(real, pred, labels=list(ALGORITMOS))
    tabla = pd.DataFrame(conteos, index=ALGORITMOS, columns=ALGORITMOS)
    tabla.index.name = "real_predicho"
    return tabla


def evaluar():
    train = pd.read_csv(RESULTADOS / "train_etiquetado.csv", float_precision="round_trip")
    with threadpool_limits(limits=1):
        config, modelo = seleccionar(train)
        selector = exportar(modelo, config, train)

        modelo_distancia, selector_distancia = ajustar_distancia(train, SEMILLA, PLIEGUES)
        guardar_json("selector_distancia.json", selector_distancia)

        # El conjunto de prueba entra después de elegir y ajustar el modelo final.
        test = pd.read_csv(RESULTADOS / "test_etiquetado.csv", float_precision="round_trip")
        y = test["best_algorithm"].to_numpy()
        pred = modelo.predict(test[columnas(config)])
        portatiles = [predecir_fila(selector, fila) for fila in test.to_dict("records")]
        if not np.array_equal(pred, portatiles):
            raise RuntimeError("El selector exportado no reproduce las predicciones del modelo.")
        pred_distancia = np.asarray(ALGORITMOS)[modelo_distancia.predict(test[ENTRADAS]).argmin(axis=1)]
        portatiles_dist = [predecir_fila(selector_distancia, fila) for fila in test.to_dict("records")]
        if not np.array_equal(pred_distancia, portatiles_dist):
            raise RuntimeError("El árbol complementario exportado difiere del árbol ajustado.")
        mayoritaria = train["best_algorithm"].value_counts().idxmax()
        base = np.repeat(mayoritaria, len(test))
        salida = test[["instance_id", "n", "distribution", *COSTOS, "best_algorithm", "n_best"]].copy()
        salida["prediccion"] = pred
        salida["baseline"] = base
        salida["costo_minimo"] = test[COSTOS].min(axis=1)
        salida["costo_elegido"] = costos_elegidos(test, pred)
        salida["costo_baseline"] = costos_elegidos(test, base)
        salida["regret"] = regret(test, pred)
        salida["regret_baseline"] = regret(test, base)
        salida["prediccion_distancia"] = pred_distancia
        salida["costo_selector_distancia"] = costos_elegidos(test, pred_distancia)
        salida["regret_distancia"] = regret(test, pred_distancia)
        salida.to_csv(RESULTADOS / "predicciones_prueba.csv", index=False)
        matriz(y, pred).to_csv(RESULTADOS / "confusion_modelo.csv")
        matriz(y, base).to_csv(RESULTADOS / "confusion_baseline.csv")
        matriz(y, pred_distancia).to_csv(RESULTADOS / "confusion_distancia.csv")
        importancia(modelo, config, test, float((pred == y).mean()))
        importancia_distancia(modelo_distancia, test, float(salida.regret_distancia.mean()))

    ambos = pd.concat([train, test], ignore_index=True)
    balance = pd.DataFrame(index=ALGORITMOS)
    for nombre, datos in (("train", train), ("test", test), ("total", ambos)):
        balance[nombre] = datos.best_algorithm.value_counts().reindex(ALGORITMOS, fill_value=0)
    balance["porcentaje_total"] = 100 * balance["total"] / len(ambos)
    balance["victorias_unicas"] = ambos.loc[ambos.n_best == 1, "best_algorithm"].value_counts().reindex(ALGORITMOS, fill_value=0)
    balance.to_csv(RESULTADOS / "balance.csv", index_label="algoritmo")

    por_grupo = pd.crosstab([ambos.n, ambos.distribution], ambos.best_algorithm).reindex(columns=ALGORITMOS, fill_value=0)
    por_grupo["total"] = por_grupo.sum(axis=1)
    por_grupo["empates_minimo"] = (ambos.n_best > 1).groupby([ambos.n, ambos.distribution]).sum()
    por_grupo.to_csv(RESULTADOS / "ganadores_por_grupo.csv")

    resumen = {"seleccion": config, "semilla_cv": SEMILLA, "modelo_base": mayoritaria,
               "instancias_train": len(train), "instancias_test": len(test),
               "empates_total": int((ambos.n_best > 1).sum()),
               "empates_test": int((test.n_best > 1).sum())}
    for nombre, etiquetas, perdidas in (("modelo", pred, salida.regret), ("baseline", base, salida.regret_baseline),
                                       ("distancia", pred_distancia, salida.regret_distancia)):
        resumen[nombre] = {"aciertos": int((etiquetas == y).sum()), "accuracy": float((etiquetas == y).mean()),
                           "regret_medio": float(perdidas.mean()), "regret_mediano": float(perdidas.median()),
                           "regret_maximo": float(perdidas.max()), "regret_cero": int((perdidas == 0).sum())}
    guardar_json("metricas.json", resumen)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
