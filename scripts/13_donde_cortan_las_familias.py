"""¿Dónde cortan los árboles en cada familia? (extensión de S4, sin regla de decisión)

09 encontró que el modelo sin corregir (F0) corta los montos en pesos sobre
todo en valores bajos: el 25% más bajo de los clientes que tienen el
producto, y a veces justo en el cero ("tiene o no tiene"). Acá se repite la
misma medición en el modelo final de cada familia (semilla 100081, entrena
201903-201908): ¿las correcciones y las proporciones también cortan abajo?

Para cada corte de un árbol sobre el nivel o un lag de una variable en pesos
(ya corregida en F1, F2 y F3), dónde cae el umbral entre los clientes del
entrenamiento que tienen ese valor con el mismo signo, con las mismas
funciones y categorías de 09. Cada corte pesa según su ganancia en el árbol.
En las proporciones (F4) y en las diferencias (F5) se miden aparte sus
propias columnas (los cocientes y las restas).

Los modelos salen de data/modelos/ (los guarda 11) o se entrenan igual que
en 04 y se guardan ahí. Control: en F0 tiene que dar lo mismo que 09.

Salida:
  output/donde_cortan_las_familias.md
  output/donde_cortan_las_familias.png

Uso:
    uv run python scripts/13_donde_cortan_las_familias.py
    uv run python scripts/13_donde_cortan_las_familias.py --metodos ninguna proporciones
"""
from __future__ import annotations

import argparse
import importlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd

from comun import cargar_meses, columnas_features, es_monetaria
from config import MES_TEST, MESES_ENTRENAMIENTO_FINAL, METODOS, NOMBRE_FAMILIA, OUTPUT_DIR, SEMILLAS

r6 = importlib.import_module("06_resumen")
r9 = importlib.import_module("09_donde_corta")  # medición, categorías y nombres de 09
r11 = importlib.import_module("11_estres_inflacion")  # modelos finales guardados en data/modelos/
decimal, tabla_md = r6.decimal, r6.tabla_md
FONDO, TINTA, TINTA_2 = r6.FONDO, r6.TINTA, r6.TINTA_2
CATEGORIAS = r9.CATEGORIAS
SEMILLA = SEMILLAS[0]
# rampa secuencial de un solo tono (de "no tiene" a montos altos): el orden importa, no la identidad
RAMPA = ["#c9daf2", "#7fa9e3", "#2a78d6", "#14478a"]


def medir(metodo: str) -> dict:
    t0 = time.time()
    feats = columnas_features(metodo, MES_TEST)
    X, k = cargar_meses(metodo, MESES_ENTRENAMIENTO_FINAL, feats)
    modelo = r11.modelos_finales(metodo, [SEMILLA], X, k, feats)[SEMILLA]
    arboles = modelo.trees_to_dataframe()
    cortes = arboles[arboles.split_feature.notna()][["split_feature", "split_gain", "threshold"]]
    total = cortes.split_gain.sum()
    cortes = cortes.assign(base=cortes.split_feature.map(lambda f: r9.separar(f)[0]),
                           tipo=cortes.split_feature.map(lambda f: r9.separar(f)[1]))
    cortes = cortes[cortes.tipo != "delta"]  # como en 09: nivel y lags; los deltas no son montos
    col = {f: j for j, f in enumerate(feats)}
    grupos = {"montos en pesos": cortes[cortes.base.map(es_monetaria)],
              "cocientes": cortes[cortes.base.str.startswith("prop_")],
              "diferencias": cortes[cortes.base.str.startswith("dif_")]}
    salida = {}
    for nombre, c in grupos.items():
        if c.empty:
            continue
        medidas = [r9.medir_cortes(g.threshold.to_numpy(float), X[:, col[f]].astype(float)).set_index(g.index)
                   for f, g in c.groupby("split_feature")]
        c = c.join(pd.concat(medidas))
        c["categoria"] = c.percentil.map(r9.categoria)
        reparto = (c.groupby("categoria").split_gain.sum() / c.split_gain.sum() * 100).reindex(CATEGORIAS).fillna(0)
        salida[nombre] = {"reparto": reparto, "pct_del_modelo": c.split_gain.sum() / total * 100}
    print(f"  {NOMBRE_FAMILIA[metodo]}: " + " | ".join(
        f"{n}: " + ", ".join(f"{decimal(v, 1)}%" for v in d["reparto"]) for n, d in salida.items())
        + f" ({time.time() - t0:.0f} s)", flush=True)
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", choices=METODOS,
                        default=["ninguna", "ipc", "mediana", "ranking", "proporciones", "diferencias"])
    metodos = parser.parse_args().metodos
    resultados = {m: medir(m) for m in metodos}

    # control: F0 tiene que dar lo mismo que 09 (mismo modelo, mismos datos, misma medición)
    if "ninguna" in resultados:
        texto_09 = (OUTPUT_DIR / "donde_corta.md").read_text(encoding="utf-8")
        for cat, v in resultados["ninguna"]["montos en pesos"]["reparto"].items():
            if f"| {cat} | {decimal(v, 1)}% |" not in texto_09:
                raise SystemExit(f"F0 no coincide con 09 en '{cat}' ({decimal(v, 1)}%)")
        print("control: en el modelo sin corregir da lo mismo que 09", flush=True)

    filas = []
    for m, r in resultados.items():
        for grupo, d in r.items():
            filas.append({"familia": NOMBRE_FAMILIA[m], "columnas": grupo,
                          "parte de la ganancia de los cortes del modelo": f"{decimal(d['pct_del_modelo'], 1)}%",
                          **{cat: f"{decimal(v, 1)}%" for cat, v in d["reparto"].items()}})
    partes = ["# ¿Dónde cortan los árboles en cada familia? (generado por scripts/13_donde_cortan_las_familias.py)\n",
              f"Extensión de S4. Modelo final de cada familia (semilla {SEMILLA}, entrena "
              f"{MESES_ENTRENAMIENTO_FINAL[0]}-{MESES_ENTRENAMIENTO_FINAL[-1]}). Cortes sobre el nivel y los lags; "
              "para cada uno, dónde cae el umbral entre los clientes del entrenamiento que tienen ese valor con el "
              "mismo signo. Cada corte pesa según su ganancia; las categorías suman 100% dentro de cada fila.\n",
              tabla_md(pd.DataFrame(filas)), ""]
    (OUTPUT_DIR / "donde_cortan_las_familias.md").write_text("\n".join(partes), encoding="utf-8")
    grafico(resultados)
    print("\n".join(partes))
    print("-> output/donde_cortan_las_familias.md y .png")


def grafico(resultados: dict) -> None:
    etiqueta = {"montos en pesos": "montos en pesos", "cocientes": "los cocientes", "diferencias": "las diferencias"}
    filas = [(NOMBRE_FAMILIA[m] + ("" if len(r) == 1 else f": {etiqueta[g]}"), d["reparto"])
             for m, r in resultados.items() for g, d in r.items()]
    alto = 2.4 + 0.7 * len(filas)
    fig, ax = plt.subplots(figsize=(11, alto))
    for i, (_, reparto) in enumerate(filas):
        izquierda = 0.0
        for j, (cat, v) in enumerate(reparto.items()):
            ax.barh(i, v, left=izquierda, color=RAMPA[j], height=0.62, edgecolor=FONDO, linewidth=2, zorder=3)
            if v >= 6:
                ax.text(izquierda + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=11,
                        color=FONDO if j >= 2 else TINTA, zorder=4)
            izquierda += v
    ax.set_yticks(range(len(filas)), [n for n, _ in filas])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("% de la ganancia de los cortes sobre esas columnas (nivel y lags)")
    ax.set_title("¿Dónde cortan los árboles en cada familia?", pad=50)
    ax.text(0, 1.025, f"Modelo final de cada familia (semilla {SEMILLA}). Dónde cae cada umbral entre los clientes\n"
            "que tienen el producto; \"separa el cero\" = pregunta si el cliente tiene o no tiene.",
            transform=ax.transAxes, fontsize=11, color=TINTA_2, va="bottom")
    fig.tight_layout(rect=(0, 0.8 / alto, 1, 1))
    fig.legend(handles=[Patch(facecolor=RAMPA[j], label=cat) for j, cat in enumerate(CATEGORIAS)],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUTPUT_DIR / "donde_cortan_las_familias.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
