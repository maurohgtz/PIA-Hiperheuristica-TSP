"""Figuras de evaluación basadas en las predicciones, las etiquetas y los tours guardados."""
import json
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .datos import ALGORITMOS, DATOS, FIGURAS, NOMBRES, RESULTADOS, leer_puntos

COLORES = ("#316d72", "#bb7836", "#69629a")
TINTA = "#24343c"
DISTRIBUCION_ES = {"uniform": "Uniforme", "clustered": "Agrupada", "border": "Bordes"}


def guardar(fig, nombre):
    fig.savefig(FIGURAS / nombre, dpi=220, facecolor="white")
    plt.close(fig)


def graficar(ejemplo):
    FIGURAS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TINTA,
                         "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False})
    matrices_confusion()
    patrones()
    rutas(ejemplo)


def matrices_confusion():
    tablas = [pd.read_csv(RESULTADOS / f"confusion_{s}.csv", index_col=0).to_numpy()
              for s in ("modelo", "distancia")]
    maximo = max(tabla.max() for tabla in tablas)
    tonos = LinearSegmentedColormap.from_list("verde", ["#f0f5f4", "#20616a"])
    etiquetas = ["2-opt", "SA", "MST"]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.85))
    for ax, tabla, titulo in zip(axes, tablas, ["Clasificador principal", "Selector por distancia"]):
        ax.imshow(tabla, cmap=tonos, vmin=0, vmax=maximo, aspect="auto")
        for i, j in np.ndindex(tabla.shape):
            ax.text(j, i, int(tabla[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold" if i == j else "normal",
                    color="white" if tabla[i, j] > 0.48 * maximo else TINTA)
        ax.set_xticks(range(3), etiquetas)
        ax.set_yticks(range(3), etiquetas)
        ax.set(xlabel="Elección", ylabel="Etiqueta real")
        ax.set_title(titulo, fontsize=12, pad=8)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.7)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=.095, right=.985, bottom=.20, top=.84, wspace=.50)
    guardar(fig, "confusion.png")


def patrones():
    grupos = pd.read_csv(RESULTADOS / "ganadores_por_grupo.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.35, 3.65), sharey=True)
    tamanos = sorted(grupos.n.unique())
    for ax, (cat, nombre) in zip(axes, DISTRIBUCION_ES.items()):
        sub = grupos[grupos.distribution == cat].set_index("n").loc[tamanos]
        y = np.arange(len(sub))
        base = np.zeros(len(sub))
        for etiqueta, color in zip(ALGORITMOS, COLORES):
            porcentaje = 100 * sub[etiqueta].to_numpy() / sub.total.to_numpy()
            ax.barh(y, porcentaje, left=base, color=color, height=.69)
            base += porcentaje
        ax.set_title(f"{nombre}  ·  {int(sub.total.sum())}", fontsize=11.5, pad=7)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0", "50", "100 %"])
        ax.set_yticks(y, [str(n) for n in tamanos])
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dce2e3", linewidth=.7)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#dce2e3")
    axes[0].set_ylabel("Ciudades (n)")
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(facecolor=c, label=NOMBRES[a]) for a, c in zip(ALGORITMOS, COLORES)],
               loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.082, right=.979, top=.875, bottom=.17, wspace=.22)
    guardar(fig, "ganadores.png")


def rutas(ejemplo):
    fila = pd.read_csv(RESULTADOS / "predicciones_prueba.csv").set_index("instance_id").loc[ejemplo]
    with ZipFile(DATOS / "instancias.zip") as archivo, archivo.open(ejemplo + ".csv") as origen:
        puntos = leer_puntos(origen)
    with (RESULTADOS / "rutas.jsonl").open() as archivo:
        recorridos = next(r["rutas"] for linea in archivo if (r := json.loads(linea))["instance_id"] == ejemplo)
    fig, axes = plt.subplots(1, 3, figsize=(8.35, 3.0))
    for ax, alg, color in zip(axes, ALGORITMOS, COLORES):
        tour = recorridos[alg]
        coords = puntos[[*tour, tour[0]]]
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=1.35, zorder=1)
        ax.scatter(puntos[:, 0], puntos[:, 1], s=11, color=TINTA, zorder=2)
        ax.set(aspect="equal", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        ax.set_xticks([0, .5, 1], ["0", ".5", "1"])
        ax.set_yticks([0, .5, 1], ["0", ".5", "1"])
        ax.tick_params(length=2.2, labelsize=8)
        for k in ("left","bottom"):
            ax.spines[k].set_color("#aab9bb")
        ax.set_title(NOMBRES[alg], fontsize=12, pad=7)
        ax.set_xlabel(f"Costo {fila['cost_' + alg]:.4f}", fontsize=10)
    fig.subplots_adjust(left=.06, right=.987, bottom=.17, top=.84, wspace=.24)
    guardar(fig, "rutas_ejemplo.png")
