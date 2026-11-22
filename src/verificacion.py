"""Verifica los tours y las métricas guardadas de forma independiente al ajuste."""
import json
import math
from itertools import permutations
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .datos import ALGORITMOS, COSTOS, DATOS, RESULTADOS, caracteristicas, leer_puntos, particiones_originales
from .modelo import guardar_json, predecir_fila


def verificar():
    partes = particiones_originales()
    tablas = {nom: pd.read_csv(RESULTADOS / f"{nom}_etiquetado.csv", float_precision="round_trip")
              for nom in partes}
    rutas = [json.loads(l) for l in (RESULTADOS / "rutas.jsonl").read_text().splitlines()]
    bruto = {fila["instance_id"]: fila for fila in rutas}
    assert len(bruto) == len(rutas) == 500
    pred = pd.read_csv(RESULTADOS / "predicciones_prueba.csv", float_precision="round_trip").set_index("instance_id")
    assert pred.index.tolist() == partes["test"].instance_id.tolist()
    selectores = {tipo: json.loads((RESULTADOS / archivo).read_text()) for tipo, archivo in
                  (("prediccion", "selector.json"), ("prediccion_distancia", "selector_distancia.json"))}
    exactos = 0
    alcanzan_optimo = np.zeros(len(ALGORITMOS), dtype=int)
    with ZipFile(DATOS / "instancias.zip") as archivo:
        for split, datos in tablas.items():
            assert datos[partes[split].columns].equals(partes[split])
            for fila in datos.itertuples(index=False):
                registro = bruto[fila.instance_id]
                assert registro["particion"] == split
                with archivo.open(f"{fila.instance_id}.csv") as f:
                    puntos = leer_puntos(f)
                esperado = list(range(fila.n))

                def longitud(ruta):
                    return math.fsum(math.dist(puntos[ruta[i - 1]], puntos[ciudad])
                                     for i, ciudad in enumerate(ruta))

                costos = np.asarray([getattr(fila, c) for c in COSTOS])
                for nombre, registrado in zip(ALGORITMOS, costos):
                    ruta = registro["rutas"][nombre]
                    assert sorted(ruta) == esperado
                    assert abs(longitud(ruta) - registrado) <= 1.1e-12
                assert fila.best_algorithm == ALGORITMOS[costos.argmin()]
                assert fila.n_best == int((costos == costos.min()).sum())
                if fila.n == 5:
                    optimo = min(longitud((0, *p)) for p in permutations(range(1, 5)))
                    assert (costos >= optimo - 1.1e-12).all()
                    alcanzan_optimo += np.isclose(costos, optimo, atol=1.1e-12, rtol=0)
                    exactos += 1
                if split == "test":
                    f = caracteristicas(puntos, fila.distribution)
                    for columna, selector in selectores.items():
                        assert predecir_fila(selector, f) == pred.at[fila.instance_id, columna]
                    assert pred.at[fila.instance_id, "best_algorithm"] == fila.best_algorithm
                    assert np.array_equal(pred.loc[fila.instance_id, COSTOS].to_numpy(dtype=float), costos)
    metricas = json.loads((RESULTADOS / "metricas.json").read_text())
    clase_base = tablas["train"].best_algorithm.value_counts().idxmax()
    assert (pred.baseline == clase_base).all()
    criterios = (("modelo", "prediccion", "regret", "costo_elegido"),
                 ("baseline", "baseline", "regret_baseline", "costo_baseline"),
                 ("distancia", "prediccion_distancia", "regret_distancia", "costo_selector_distancia"))
    for clave, eleccion, perdida, columna_costo in criterios:
        cm = pd.crosstab(pred.best_algorithm, pred[eleccion]).reindex(index=ALGORITMOS, columns=ALGORITMOS, fill_value=0)
        guardada = pd.read_csv(RESULTADOS / f"confusion_{clave}.csv", index_col=0)
        assert np.array_equal(cm.to_numpy(), guardada.loc[list(ALGORITMOS), list(ALGORITMOS)].to_numpy())
        elegidos = np.array([row["cost_" + row[eleccion]] for _, row in pred.iterrows()])
        minimos = pred[COSTOS].min(axis=1).to_numpy()
        diferencias = elegidos - minimos
        assert np.array_equal(elegidos, pred[columna_costo].to_numpy())
        assert np.array_equal(minimos, pred.costo_minimo.to_numpy())
        assert np.allclose(diferencias, pred[perdida], atol=1e-12, rtol=0)
        assert np.all(diferencias >= 0)
        assert np.isclose(diferencias.mean(), metricas[clave]["regret_medio"])
        assert np.isclose((pred.best_algorithm == pred[eleccion]).mean(), metricas[clave]["accuracy"])
    guardar_json("controles.json", {"instancias": len(rutas), "rutas_verificadas": 3 * len(rutas),
                                    "exactos_n5": exactos, "aciertos_exactos_n5": dict(zip(ALGORITMOS, alcanzan_optimo.tolist()))})
    print(f"Verificación correcta. 500 instancias, 1500 rutas, {exactos} óptimos exactos de cinco ciudades y 149 predicciones por selector.")
