"""Corre los tres algoritmos de la Tarea 2 sobre la partición oficial."""
import json
import math
import random
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .algoritmos import dos_opt, matriz_distancias, mst_preorden, recocido_simulado, vecino_mas_cercano
from .datos import ALGORITMOS, COSTOS, DATOS, EXTRAS, MEDIDAS, RESULTADOS, caracteristicas, leer_puntos, particiones_originales

# Equivalentes a 40 y 0.1 en el plano [0,100] de la Tarea 2.
T_INICIAL = 0.40
T_FINAL = 0.001
ITERACIONES = 4500


def resolver(puntos, indice):
    n = len(puntos)
    dist = matriz_distancias(puntos.tolist())
    ciudad_inicial = random.Random(40000 + indice).randrange(n)
    inicio_local = vecino_mas_cercano(dist, ciudad_inicial)
    inicio_sa = random.Random(6000 + indice).sample(range(n), n)
    rutas = {
        "2opt_hill_climbing": dos_opt(inicio_local, dist)["ruta"],
        "sa_2opt": recocido_simulado(inicio_sa, dist, esquema="exponencial",
                                     iteraciones=ITERACIONES, temp_inicial=T_INICIAL,
                                     temp_final=T_FINAL, semilla=10000 + indice)["ruta"],
        "mst_2opt": dos_opt(mst_preorden(dist), dist)["ruta"],
    }

    costos = []
    for nombre in ALGORITMOS:
        ruta = rutas[nombre]
        if sorted(ruta) != list(range(n)):
            raise RuntimeError(f"Recorrido inválido en instancia {indice}, {nombre}.")
        total = math.fsum(dist[ruta[k - 1]][ruta[k]] for k in range(n))
        costos.append(round(total, 12))

    minimo = min(costos)
    ganadores = [a for a, costo in zip(ALGORITMOS, costos) if costo == minimo]
    return costos, ganadores, rutas


def etiquetar():
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    partes = particiones_originales()
    with ZipFile(DATOS / "instancias.zip") as archivos:
        identificadores = [x for df in partes.values() for x in df.instance_id]
        if sorted(archivos.namelist()) != sorted(f"{x}.csv" for x in identificadores):
            raise ValueError("Las coordenadas no coinciden con los identificadores de train y test.")
        auditoria = []
        for split, tabla in partes.items():
            filas = []
            for fila in tabla.itertuples(index=False):
                with archivos.open(f"{fila.instance_id}.csv") as documento:
                    puntos = leer_puntos(documento)
                if len(puntos) != fila.n or (puntos < 0).any() or (puntos > 1).any():
                    raise ValueError(f"Tamaño o escala incorrecta en {fila.instance_id}.")
                calculadas = caracteristicas(puntos, fila.distribution)
                if not all(np.isclose(calculadas[c], getattr(fila, c), rtol=1e-10, atol=1e-12)
                           for c in MEDIDAS):
                    raise ValueError(f"Las características no corresponden a {fila.instance_id}.")
                indice = int(fila.instance_id.split("_")[1])
                costos, ganadores, rutas = resolver(puntos, indice)
                fila_extra = {c: calculadas[c] for c in EXTRAS}
                fila_extra.update(zip(COSTOS, costos))
                fila_extra.update(best_algorithm=ganadores[0], n_best=len(ganadores))
                filas.append(fila_extra)
                auditoria.append({"instance_id": fila.instance_id, "particion": split, "rutas": rutas})
            tabla = pd.concat([tabla, pd.DataFrame(filas, index=tabla.index)], axis=1)
            tabla.to_csv(RESULTADOS / f"{split}_etiquetado.csv", index=False)
            print(f"{split}: {len(tabla)} instancias etiquetadas.", flush=True)

    with (RESULTADOS / "rutas.jsonl").open("w", encoding="utf-8") as salida:
        for registro in auditoria:
            salida.write(json.dumps(registro, separators=(",", ":")) + "\n")
