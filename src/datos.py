"""Lectura de instancias y cálculo de las características definidas en el curso."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RAIZ / "figuras"
DISTRIBUCIONES = ("uniform", "clustered", "border")
MEDIDAS = ("mean_pairwise_dist", "std_coords", "mean_centroid_dist", "mean_nn_dist")
EXTRAS = ("cv_nn", "fraccion_casco")
ATRIBUTOS = ("n", "distribution", *MEDIDAS)
ALGORITMOS = ("2opt_hill_climbing", "sa_2opt", "mst_2opt")
COSTOS = ["cost_" + nombre for nombre in ALGORITMOS]
NOMBRES = dict(zip(ALGORITMOS, ("2-opt", "SA + 2-opt", "MST + 2-opt")))


def leer_puntos(origen):
    tabla = pd.read_csv(origen, float_precision="round_trip")
    if not {"x", "y"}.issubset(tabla.columns):
        raise ValueError("El CSV de coordenadas debe incluir x e y.")
    puntos = tabla.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if len(puntos) < 4 or not np.isfinite(puntos).all():
        raise ValueError("Se requieren al menos cuatro puntos con coordenadas finitas.")
    return puntos


def vertices_casco(puntos):
    """Número de vértices extremos por cadena monótona, sin colineales intermedios."""
    ordenados = sorted(set(map(tuple, puntos)))
    if len(ordenados) <= 2:
        return len(ordenados)

    def cadena(secuencia):
        parte = []
        for punto in secuencia:
            while len(parte) >= 2:
                a, b = parte[-2:]
                cruz = (b[0] - a[0]) * (punto[1] - a[1]) - (b[1] - a[1]) * (punto[0] - a[0])
                if cruz > 0:
                    break
                parte.pop()
            parte.append(punto)
        return parte

    return len(cadena(ordenados)) + len(cadena(reversed(ordenados))) - 2


def caracteristicas(puntos, distribucion):
    if distribucion not in DISTRIBUCIONES:
        raise ValueError(f"Distribución válida: {', '.join(DISTRIBUCIONES)}")
    n = len(puntos)
    distancias = np.linalg.norm(puntos[:, None] - puntos[None, :], axis=2)
    pares = distancias[np.triu_indices(n, k=1)]
    np.fill_diagonal(distancias, math.inf)
    vecinos = distancias.min(axis=1)
    media_vecino = float(vecinos.mean())
    return {
        "n": n,
        "distribution": distribucion,
        "mean_pairwise_dist": float(pares.mean()),
        "std_coords": float(puntos.std(axis=0, ddof=0).mean()),
        "mean_centroid_dist": float(np.linalg.norm(puntos - puntos.mean(axis=0), axis=1).mean()),
        "mean_nn_dist": media_vecino,
        "cv_nn": float(vecinos.std(ddof=0) / media_vecino) if media_vecino > 0 else 0.0,
        "fraccion_casco": vertices_casco(puntos) / n,
    }


def particiones_originales():
    partes = {nombre: pd.read_csv(DATOS / f"{nombre}.csv", float_precision="round_trip")
              for nombre in ("train", "test")}
    juntos = pd.concat(partes.values(), ignore_index=True)
    if [len(df) for df in partes.values()] != [351, 149] or not juntos.instance_id.is_unique:
        raise ValueError("La partición del curso requiere 351 y 149 identificadores distintos.")
    for nombre, df in partes.items():
        if tuple(df.columns) != ("instance_id", *ATRIBUTOS) or df.isna().any().any():
            raise ValueError(f"Esquema de datos incorrecto en {nombre}.")
        if not df.instance_id.str.fullmatch(r"inst_[0-9]{4}").all():
            raise ValueError("Identificador de instancia inesperado.")
        if set(df.distribution) != set(DISTRIBUCIONES) or df.groupby(["n", "distribution"]).ngroups != 27:
            raise ValueError("La partición debe contener las 27 combinaciones de tamaño y distribución.")
    return partes
