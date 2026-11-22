"""Proyecto final de Algoritmia y Optimización. Ejecutar desde la carpeta del proyecto."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from src.datos import DATOS, DISTRIBUCIONES, NOMBRES, RESULTADOS, caracteristicas, leer_puntos, particiones_originales


def predecir(args, parser):
    from src.modelo import predecir_fila
    if args.instancia:
        partes = particiones_originales()
        categorias = {r.instance_id: r.distribution for tabla in partes.values() for r in tabla.itertuples()}
        if args.instancia not in categorias:
            parser.error("El identificador no aparece en el dataset.")
        distribucion = categorias[args.instancia]
        if args.distribucion and args.distribucion != distribucion:
            parser.error("La distribución indicada difiere de la registrada para esa instancia.")
        with ZipFile(DATOS / "instancias.zip") as archivo, archivo.open(args.instancia + ".csv") as f:
            puntos = leer_puntos(f)
    else:
        if not args.distribucion:
            parser.error("Indica --distribucion para un CSV externo.")
        distribucion = args.distribucion
        puntos = leer_puntos(args.csv)
    if (puntos < 0).any() or (puntos > 1).any():
        parser.error("El modelo se entrenó con coordenadas en [0, 1]. Usa esa escala.")
    nombre = "selector.json" if args.criterio == "acierto" else "selector_distancia.json"
    selector = json.loads((RESULTADOS / nombre).read_text(encoding="utf-8"))
    eleccion = predecir_fila(selector, caracteristicas(puntos, distribucion))
    print(f"{len(puntos)} ciudades · {distribucion} · criterio de {args.criterio}")
    print(f"Algoritmo elegido: {NOMBRES[eleccion]} ({eleccion})")


def main():
    parser = argparse.ArgumentParser(description="Hiperheurística de selección para TSP")
    sub = parser.add_subparsers(dest="accion")
    sub.add_parser("etiquetar", help="Ejecuta los tres algoritmos sobre train y test.")
    sub.add_parser("entrenar", help="Ajusta y evalúa los modelos con los costos guardados.")
    sub.add_parser("verificar", help="Comprueba rutas, etiquetas, predicciones y métricas.")
    rep = sub.add_parser("reporte", help="Genera figuras y PDF a partir de los resultados.")
    rep.add_argument("--repo", help="Enlace público de GitHub o GitLab para incluir en el PDF.")
    pred = sub.add_parser("predecir", help="Selecciona un algoritmo para una instancia.")
    fuente = pred.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--csv", type=Path, help="CSV de dos columnas x,y.")
    fuente.add_argument("--instancia", help="Identificador del conjunto incluido.")
    pred.add_argument("--distribucion", choices=DISTRIBUCIONES)
    pred.add_argument("--criterio", choices=["acierto", "distancia"], default="acierto")
    args = parser.parse_args()
    if args.accion == "predecir":
        predecir(args, parser)
        return
    if args.accion in (None, "etiquetar"):
        from src.etiquetado import etiquetar
        etiquetar()
    if args.accion in (None, "entrenar"):
        from src.modelo import evaluar
        evaluar()
    if args.accion in (None, "verificar"):
        from src.verificacion import verificar
        verificar()
    if args.accion in (None, "reporte"):
        from src.informe import generar_informe
        generar_informe(getattr(args, "repo", None))


if __name__ == "__main__":
    main()
