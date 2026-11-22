"""Reporte del experimento. Lee las tablas y los tours producidos por el proyecto."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from .datos import ALGORITMOS, FIGURAS, NOMBRES, RAIZ, RESULTADOS
from .figuras import DISTRIBUCION_ES, graficar

ANCHO, ALTO = letter
MARGEN = 48
CONTENIDO = ANCHO - 2 * MARGEN
TINTA = "#20343B"
VERDE = "#235E64"
GRIS = "#4D5F65"


def json_resultado(archivo):
    return json.loads((RESULTADOS / archivo).read_text(encoding="utf-8"))


def fuente_pdf():
    carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for alias, archivo in (("Texto", "DejaVuSans.ttf"), ("TextoB", "DejaVuSans-Bold.ttf"),
                           ("Titular", "DejaVuSerif.ttf")):
        pdfmetrics.registerFont(TTFont(alias, str(carpeta / archivo)))
    pdfmetrics.registerFontFamily("Texto", normal="Texto", bold="TextoB")


def estilo(nombre, fuente, puntos, interlineado, color, alineacion=TA_LEFT):
    return ParagraphStyle(nombre, fontName=fuente, fontSize=puntos, leading=interlineado,
                          textColor=colors.HexColor(color), alignment=alineacion)


class Paginas:
    def __init__(self, destino):
        fuente_pdf()
        self.c = Canvas(str(destino), pagesize=letter, pageCompression=1)
        self.c.setTitle("Hiperheurística de selección para el problema del viajero")
        self.c.setAuthor("Mauro Herrera")
        self.c.setSubject("Algoritmia y Optimización · Proyecto final")
        self.y = 0
        self.numero = 0

    def iniciar(self, titulo, primera=False):
        if self.numero:
            self.terminar_pagina()
        self.numero += 1
        self.c.setFillColor(colors.HexColor(VERDE))
        self.c.setFont("TextoB", 8.6)
        self.c.drawString(MARGEN, ALTO - 36, "Algoritmia y Optimización")
        self.c.setFont("Texto", 8.6)
        self.c.drawRightString(ANCHO - MARGEN, ALTO - 36, "Proyecto final   /   PIA")
        self.c.setStrokeColor(colors.HexColor("#AEC1C0"))
        self.c.setLineWidth(.65)
        self.c.line(MARGEN, ALTO - 45, ANCHO - MARGEN, ALTO - 45)
        self.y = ALTO - 62
        tipo = estilo("Título", "Titular", 26 if primera else 23, 31 if primera else 29, TINTA)
        self.texto(titulo, tipo, 8)

    def texto(self, contenido, tipo, espacio):
        p = Paragraph(contenido, tipo)
        _, altura = p.wrap(CONTENIDO, ALTO)
        p.drawOn(self.c, MARGEN, self.y - altura)
        self.y -= altura + espacio

    def p(self, contenido, espacio=8):
        self.texto(contenido, estilo("Párrafo", "Texto", 10, 14.5, TINTA), espacio)

    def nota(self, contenido, espacio=8):
        self.texto(contenido, estilo("Nota", "Texto", 8.7, 12.5, GRIS), espacio)

    def h(self, contenido):
        self.y -= 3
        self.texto(contenido, estilo("Subtítulo", "TextoB", 12.4, 16, VERDE), 6)

    def tabla(self, cab, filas, anchos, numeros=(), destacado=(), puntos=9.1):
        if abs(sum(anchos) - CONTENIDO) > .01:
            raise ValueError("El ancho de la tabla debe coincidir con el del texto.")
        contenido = []
        for i, fila in enumerate([cab, *filas]):
            celdas = []
            for j, valor in enumerate(fila):
                negrita = i == 0 or (i - 1 in destacado and j == 0)
                tipo = estilo("Celda", "TextoB" if negrita else "Texto", puntos - .4 if i == 0 else puntos,
                              12.4, VERDE if i == 0 else TINTA, TA_RIGHT if j in numeros else TA_LEFT)
                celdas.append(Paragraph(str(valor), tipo))
            contenido.append(celdas)
        t = Table(contenido, colWidths=anchos)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9F9")]),
            ("LINEBELOW", (0, 0), (-1, 0), .65, colors.HexColor("#9AB4B4")),
            ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#B9C6C7")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        _, altura = t.wrap(CONTENIDO, ALTO)
        t.drawOn(self.c, MARGEN, self.y - altura)
        self.y -= altura + 8

    def figura(self, archivo, ancho=CONTENIDO):
        img = ImageReader(str(FIGURAS / archivo))
        w, h = img.getSize()
        altura = ancho * h / w
        self.c.drawImage(img, (ANCHO - ancho) / 2, self.y - altura, width=ancho, height=altura)
        self.y -= altura + 3

    def terminar_pagina(self):
        if self.y < 46:
            raise ValueError(f"Contenido por debajo del margen en página {self.numero}: {self.y:.1f}")
        self.c.setStrokeColor(colors.HexColor("#BAC8C8"))
        self.c.line(MARGEN, 39, ANCHO - MARGEN, 39)
        self.c.setFillColor(colors.HexColor(GRIS))
        self.c.setFont("Texto", 8)
        self.c.drawString(MARGEN, 25, "Mauro Herrera   ·   Selección de heurísticas para TSP")
        self.c.drawRightString(ANCHO - MARGEN, 25, f"{self.numero} / 5")
        self.c.showPage()

    def guardar(self):
        self.terminar_pagina()
        self.c.save()


def generar_informe(repo=None):
    ubicacion = RAIZ / "repositorio.txt"
    if repo:
        url = urlsplit(repo.strip().removesuffix(".git").rstrip("/"))
        segmentos = [s for s in url.path.split("/") if s]
        if url.scheme != "https" or url.netloc not in ("github.com", "gitlab.com") or len(segmentos) < 2:
            raise ValueError("Indica la URL https del repositorio de GitHub o GitLab.")
        ubicacion.write_text(f"https://{url.netloc}{url.path}\n", encoding="utf-8")
    enlace = ubicacion.read_text(encoding="utf-8").strip() if ubicacion.exists() else None
    m = json_resultado("metricas.json")
    controles = json_resultado("controles.json")
    principal, base, alternativa = (m[c] for c in ("modelo", "baseline", "distancia"))
    seleccion = m["seleccion"]
    config_dist = json_resultado("selector_distancia.json")["configuracion"]
    balance = pd.read_csv(RESULTADOS / "balance.csv").set_index("algoritmo")
    val = pd.read_csv(RESULTADOS / "validacion_cruzada.csv")
    val_dist = pd.read_csv(RESULTADOS / "validacion_distancia.csv")
    tr = pd.read_csv(RESULTADOS / "train_etiquetado.csv")
    te = pd.read_csv(RESULTADOS / "test_etiquetado.csv")
    df = pd.read_csv(RESULTADOS / "predicciones_prueba.csv")
    grupos = pd.read_csv(RESULTADOS / "ganadores_por_grupo.csv")
    todos = pd.concat([tr, te])
    cv = val.iloc[0]
    ejemplo = df.loc[(df.regret - df.regret_distancia).idxmax()]
    peor = df.loc[df.regret.idxmax()]
    graficar(ejemplo.instance_id)
    destino = RAIZ / "PIA_Hiperheuristica_TSP.pdf"
    doc = Paginas(destino)
    doc.iniciar("Hiperheurística de selección<br/>para el problema del viajero", primera=True)
    doc.nota("Mauro Herrera  ·  22 de noviembre de 2026", 11)
    doc.p("Dadas las coordenadas de una instancia del TSP, ¿se puede anticipar cuál de tres heurísticas producirá el circuito más corto? Para construir las etiquetas se ejecutó cada algoritmo sobre las 500 instancias del curso. Cada costo es la longitud euclidiana del recorrido cerrado, incluido el regreso a la primera ciudad.")
    doc.h("Balance de las etiquetas")
    rows = [[NOMBRES[a], int(r.train), int(r.test), int(r.total), f"{r.porcentaje_total:.1f} %", int(r.victorias_unicas)]
            for a, r in balance.iterrows()]
    doc.tabla(["Algoritmo", "Train", "Test", "Total", "% de 500", "Victorias únicas"], rows,
              [127, 57, 56, 55, 83, 138], numeros=(1, 2, 3, 4, 5))
    dobles, triples = [int((todos.n_best == k).sum()) for k in (2, 3)]
    doc.p(f"Hubo {m['empates_total']} empates en el mínimo ({dobles} dobles y {triples} triples). Se compararon costos redondeados a 12 decimales y se asignó una sola etiqueta en el orden 2-opt, SA y MST. Así, 2-opt recibió {int(balance.iloc[0].total)} etiquetas y fue el único ganador en {int(balance.iloc[0].victorias_unicas)} instancias.")
    doc.h("Cómo se obtuvo cada recorrido")
    doc.tabla(["Método", "Construcción y búsqueda"], [
        ["2-opt local", "Vecino más cercano desde una ciudad sorteada. En cada iteración se invierte el segmento con la mayor reducción hasta agotar las mejoras."],
        ["SA + 2-opt", "Recocido simulado desde una permutación aleatoria. Prueba inversiones 2-opt, acepta un empeoramiento Δ con probabilidad exp(−Δ/T) y conserva la mejor ruta visitada."],
        ["MST + 2-opt", "Prim desde la ciudad 0, recorrido en preorden del árbol de expansión mínima y cierre del circuito. Después se aplica el mismo descenso local 2-opt."],
    ], [101, 415], puntos=9.0)
    doc.nota("Los tres métodos reutilizan el código de la Tarea 2. Para el índice numérico i del archivo se fijaron las semillas 40000 + i (inicio local), 6000 + i (permutación de SA) y 10000 + i (movimientos de SA). MST usa desempates deterministas.")
    doc.p("El calendario exponencial de SA tiene 4,500 iteraciones y sigue T(t) = 0.40 × (0.001 / 0.40)<super>t/4499</super>. El cambio de escala de [0, 100] a [0, 1] divide entre 100 tanto las distancias como las temperaturas originales (40 y 0.1), y conserva el cociente Δ/T.")
    if enlace:
        doc.nota(f'Repositorio público · <link href="{escape(enlace)}" color="{VERDE}">{escape(enlace)}</link>')

    doc.iniciar("Entrenamiento del selector")
    doc.p(f"Se conservó la partición provista de {len(tr)} instancias para entrenar y {len(te)} para probar. Ambos grupos contienen las 27 combinaciones de nueve tamaños y tres distribuciones. Cada instancia intervino en una sola partición.")
    doc.h("Variables disponibles al predecir")
    doc.p("Se usaron el número de ciudades, la distribución (uniforme, agrupada o de bordes) y cuatro medidas del curso: la distancia media entre pares, el promedio de las desviaciones de x e y, la distancia media al centroide y la distancia media al vecino más cercano. La distribución se representó con tres indicadores binarios.")
    doc.p("También se probó un conjunto ampliado con la fracción de puntos en el casco convexo y el coeficiente de variación de las distancias al vecino más cercano. Todas las variables se calculan antes de resolver el TSP. El identificador, los tres costos y la etiqueta se excluyen de los predictores.")
    doc.h("Validación dentro del entrenamiento")
    doc.tabla(["Clasificador", "Configuraciones ensayadas", "Preparación"], [
        ["Árbol CART", "Profundidad 1, 2, 3 o 5. Mínimo de 5 o 15 observaciones por hoja. Criterio de Gini.", "Variables en su escala original."],
        ["KNN", "7, 15 o 31 vecinos. Distancia euclidiana y voto uniforme.", "Numéricas estandarizadas en cada pliegue."],
    ], [88, 252, 176])
    doc.p(f"Se compararon {len(val)} configuraciones en cinco pliegues estratificados por etiqueta, con semilla 20261122. Se eligió la mayor accuracy fuera de pliegue, con menor regret medio y orden de la cuadrícula como desempates. El modelo ganador se volvió a ajustar sobre las {len(tr)} instancias.")
    doc.h("La regla que aprendió el modelo")
    if seleccion["familia"] == "CART":
        doc.p(f"Ganó CART con los rasgos del {'curso' if seleccion['rasgos'] == 'curso' else 'conjunto ampliado'}, profundidad máxima {seleccion['max_depth']} y al menos {seleccion['min_samples_leaf']} instancias por hoja. Acertó {int(cv.aciertos_cv)} de {len(tr)} en validación ({100*cv.accuracy_cv:.2f} %).")
    else:
        doc.p(f"Ganó KNN con k = {seleccion['k']} y {100*cv.accuracy_cv:.2f} % de accuracy en validación.")
    arbol = json_resultado("selector.json")
    if seleccion["familia"] == "CART" and len(arbol["izq"]) == 3 and arbol["atributo"][0] == 0:
        umbral = arbol["umbral"][0]
        hojas = []
        for etiqueta, datos in ((f"n ≤ {umbral:g}", tr[tr.n <= umbral]), (f"n > {umbral:g}", tr[tr.n > umbral])):
            hojas.append([etiqueta, len(datos), *[int((datos.best_algorithm == a).sum()) for a in ALGORITMOS],
                          NOMBRES[datos.best_algorithm.value_counts().idxmax()]])
        doc.tabla(["Hoja", "Casos", "2-opt", "SA", "MST", "Predicción"], hojas,
                  [84, 65, 68, 65, 65, 169], numeros=(1, 2, 3, 4))
    imp = pd.read_csv(RESULTADOS / "importancias.csv")
    if (imp.caida_pp.abs() < 1e-12).all():
        doc.p("El tamaño concentra toda la reducción de impureza del árbol. La frecuencia de las clases cambia entre hojas, pero 2-opt conserva la mayoría en ambas. Al permutar cada uno de los seis atributos 40 veces en prueba, la variación media de accuracy fue 0.00 puntos porcentuales para todos.")
    else:
        top = imp.iloc[0]
        doc.p(f"Por permutación en prueba (40 repeticiones), el mayor descenso medio de accuracy correspondió a {top.atributo} con {top.caida_pp:.2f} puntos porcentuales.")

    doc.iniciar("Resultados en las 149 instancias de prueba")
    doc.p(f"El baseline elige 2-opt en todos los casos por ser la clase más frecuente de entrenamiento ({int(balance.iloc[0].train)}/{len(tr)}). La accuracy cuenta coincidencias con la etiqueta asignada. El regret de una instancia se calcula restando al costo elegido el menor de los tres costos observados.")
    rows = []
    for etiqueta, met in (("Baseline (2-opt)", base), ("Clasificador CART", principal), ("Por distancia¹", alternativa)):
        rows.append([etiqueta, f"{met['aciertos']}/{len(te)}", f"{100*met['accuracy']:.2f} %", f"{met['regret_medio']:.5f}",
                     f"{met['regret_maximo']:.5f}", f"{met['regret_cero']}/{len(te)}"])
    doc.tabla(["Método", "Aciertos", "Accuracy", "Regret medio", "Regret máx.", "Costo mínimo"], rows,
              [114, 62, 77, 88, 85, 90], numeros=(1, 2, 3, 4, 5), destacado=(1,))
    doc.nota("El costo mínimo cuenta cualquier elección con regret cero, incluidos los empates. Los tres métodos tuvieron mediana de regret igual a cero. ¹El criterio por distancia se detalla en la página 5.")
    doc.h("Matrices de confusión")
    doc.figura("confusion.png", 490)
    doc.nota("Figura 1. Las filas son etiquetas reales y las columnas son elecciones. La matriz izquierda corresponde tanto al clasificador principal como al baseline. SA y MST incluyen 2-opt.", 10)
    cm = pd.read_csv(RESULTADOS / "confusion_modelo.csv", index_col=0)
    cm_dist = pd.read_csv(RESULTADOS / "confusion_distancia.csv", index_col=0)
    nsa, nmst = int(cm.loc[ALGORITMOS[1]].sum()), int(cm.loc[ALGORITMOS[2]].sum())
    doc.p(f"El clasificador igualó al baseline en las {len(te)} decisiones. Detectó {int(cm.loc[ALGORITMOS[1], ALGORITMOS[1]])} de {nsa} etiquetas de SA y {int(cm.loc[ALGORITMOS[2], ALGORITMOS[2]])} de {nmst} de MST. Las {nsa+nmst} instancias que 2-opt perdió tuvieron regret positivo. El criterio por distancia sí identificó {int(cm_dist.loc[ALGORITMOS[1], ALGORITMOS[1]])} de las {nsa} etiquetas de SA y {int(cm_dist.loc[ALGORITMOS[2], ALGORITMOS[2]])} de las {nmst} de MST.")
    doc.h("El error de mayor costo")
    doc.p(f"En {peor.instance_id} ({int(peor.n)} ciudades, distribución de {DISTRIBUCION_ES[peor.distribution].lower()}), 2-opt recorrió {peor.costo_elegido:.5f} unidades y {NOMBRES[peor.best_algorithm]} obtuvo {peor.costo_minimo:.5f}. La diferencia de {peor.regret:.5f} explica {100*peor.regret/df.regret.sum():.1f} % del regret acumulado del clasificador.")

    doc.iniciar("Ganadores por tamaño y distribución")
    doc.p("Cada barra representa las 18 o 19 instancias con el mismo tamaño y distribución. Los porcentajes se calculan dentro de cada combinación. La figura reúne entrenamiento y prueba para describir el conjunto completo una vez evaluados los modelos.")
    doc.figura("ganadores.png")
    doc.nota("Figura 2. Proporción de etiquetas asignadas a cada algoritmo. El número en el encabezado de cada panel indica el total de instancias de esa distribución.")
    rows = []
    for cat, nombre in DISTRIBUCION_ES.items():
        parte = grupos[grupos.distribution == cat]
        total = int(parte.total.sum())
        rows.append([nombre, total, *[f"{int(parte[a].sum())} ({100*parte[a].sum()/total:.1f} %)" for a in ALGORITMOS]])
    doc.tabla(["Distribución", "Total", "2-opt", "SA + 2-opt", "MST + 2-opt"], rows,
              [112, 59, 114, 112, 119], numeros=(1, 2, 3, 4))
    n5 = controles["exactos_n5"]
    exactos = controles["aciertos_exactos_n5"]
    doc.p(f"<b>Cinco ciudades.</b> Las {n5} etiquetas fueron para 2-opt. La enumeración de los 24 recorridos con la primera ciudad fija confirmó que 2-opt y SA alcanzaron el óptimo en {exactos[ALGORITMOS[0]]} de {n5} instancias cada uno. MST lo alcanzó en {exactos[ALGORITMOS[2]]}. La etiqueta, por sí sola, oculta esas coincidencias.")
    por_n = grupos.groupby("n")[list(ALGORITMOS) + ["total"]].sum()
    doc.p(f"<b>Tamaño.</b> SA concentró {int(por_n.loc[[10, 15, 20], ALGORITMOS[1]].sum())} de sus {int(balance.loc[ALGORITMOS[1], 'total'])} etiquetas entre 10 y 20 ciudades, y no ganó a partir de 60 con este presupuesto. MST pasó a {int(por_n.loc[80, ALGORITMOS[2]])} de {int(por_n.loc[80, 'total'])} a n = 80 y bajó a {int(por_n.loc[100, ALGORITMOS[2]])} de {int(por_n.loc[100, 'total'])} a n = 100.")
    doc.p("<b>Distribución.</b> En los bordes, 2-opt acumuló la mayor proporción de etiquetas. SA fue más frecuente en puntos agrupados y MST alcanzó proporciones similares en agrupadas y uniformes. Estos patrones describen grupos. Dentro de un mismo grupo pueden ganar métodos distintos.")

    doc.iniciar("Elegir según la distancia que se pierde")
    doc.p("Ante la predicción constante de CART, se examinó un criterio complementario. Para cada instancia de entrenamiento se calculó un vector con los tres regrets. Un árbol de regresión predice esos tres valores a partir de las seis características del curso y elige el algoritmo de menor pérdida esperada.")
    doc.p(f"Se compararon {len(val_dist)} combinaciones de profundidad (1 a 4) y tamaño mínimo de hoja (10, 20 o 35) en los mismos cinco pliegues de entrenamiento. Por menor regret de validación ({val_dist.iloc[0].regret_cv:.5f}) se eligió profundidad {config_dist['max_depth']} y mínimo de {config_dist['min_samples_leaf']} por hoja. La comparación en la misma partición de prueba tiene carácter exploratorio.")
    cambio = df.regret_distancia - df.regret
    reduccion = 100 * (1 - alternativa["regret_medio"] / base["regret_medio"])
    doc.h("Un mismo error de etiqueta puede costar cero")
    doc.p(f"En prueba, el criterio por distancia redujo el regret medio un {reduccion:.1f} %. Dio una ruta más corta en {(cambio < -1e-12).sum()} casos, más larga en {(cambio > 1e-12).sum()} e igual en {(cambio.abs() <= 1e-12).sum()}. Acertó {alternativa['aciertos']} etiquetas y alcanzó el costo mínimo en {alternativa['regret_cero']} instancias. La diferencia de {alternativa['regret_cero']-alternativa['aciertos']} corresponde a empates adjudicados a otro algoritmo.")
    impd = pd.read_csv(RESULTADOS / "importancias_distancia.csv").iloc[0]
    doc.p(f"Al permutar el tamaño en prueba, el regret medio de este selector aumentó {impd.aumento_regret:.3f} unidades. Al permutar cualquiera de las otras cinco entradas se mantuvo el mismo costo medio. Este resultado mide la dependencia de las elecciones del árbol ajustado.")
    doc.figura("rutas_ejemplo.png", 500)
    doc.nota(f"Figura 3. Tres circuitos de {ejemplo.instance_id}, con {int(ejemplo.n)} ciudades y distribución {DISTRIBUCION_ES[ejemplo.distribution].lower()}. Esta instancia tuvo la mayor reducción individual de regret al cambiar de criterio.")
    doc.p(f"En ese caso, el clasificador eligió 2-opt y recorrió {ejemplo.costo_elegido:.4f}. El selector por distancia eligió {NOMBRES[ejemplo.prediccion_distancia]} y recorrió {ejemplo.costo_selector_distancia:.4f}. La diferencia fue de {ejemplo.regret-ejemplo.regret_distancia:.4f} unidades.")
    doc.h("Alcance del experimento")
    doc.p("Cada etiqueta refleja una ejecución por algoritmo, con las semillas y el presupuesto indicados. El costo de referencia es el mejor observado entre los tres recorridos. El resultado en prueba favorece criterios distintos según se busque acertar la etiqueta o reducir la distancia adicional.")
    doc.nota("Fuentes y reproducción. Coordenadas y particiones de hh_tsp_dataset, material del curso. Implementaciones de TSP de la Tarea 2, ejercicios 3 a 5. El comando <b>py main.py</b> reproduce las etiquetas, los modelos, las verificaciones y el informe.")
    doc.guardar()
    print(f"Reporte: {destino}")
