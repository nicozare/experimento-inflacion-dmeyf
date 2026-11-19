"""S9. El modelo de 2019 que envejece (barrida de meses, 19/11; regla en el README).

Los modelos finales de las cuatro familias del experimento principal, entrenados
con 201903-201908 como en 04 (hiperparámetros y árboles de su bayesiana), se
aplican sin reentrenar a cada mes de 201910 a 202106. La inflación acumulada
contra el mes promedio del entrenamiento va de +18% a +117%: es la prueba de
estrés (11) con datos reales. En el medio cambian otras cosas (la pandemia),
pero les pasan a todas las familias por igual, y la comparación es pareada por
mes y semilla. 5 semillas (config.SEMILLAS_BARRIDA).

Necesita los datos extendidos (DATOS=extendido: 01 y 02 arman 201901-202106 en
data/extendido/). Los modelos salen de data/modelos/ (los guardan 11 y 13) o se
entrenan como en 04 y se guardan ahí.

Controles:
  - los meses 201903-201910 de data/extendido/fe/ son iguales a los de data/fe/
    (mismas columnas, mismos clientes, mismos valores);
  - en 201910, la ganancia de cada semilla con el corte de la bayesiana es la de 04.

Exploratorio, agregado el 19/11 al ver los resultados: el corte fijo de la
bayesiana confunde la regla, porque en 2021 hay casi el doble de BAJA+2 y el
mejor corte se duplica (gana sola la familia que estimula más clientes). El
resumen agrega el AUC y la ganancia con el mejor corte de cada semilla, que no
dependen de ese corte, por tramos de inflación acumulada; el gráfico tiene un
segundo panel con el AUC.

Salida:
  resultados/barrida/envejece.csv          ganancia por familia, semilla y mes
  resultados/barrida/envejece_curvas.csv   curva de ganancia media de las semillas, por familia y mes
  output/barrida/envejece.md y .png

Uso:
    DATOS=extendido uv run python scripts/14_modelo_que_envejece.py
    DATOS=extendido uv run python scripts/14_modelo_que_envejece.py --solo-resumen
    DATOS=extendido uv run python scripts/14_modelo_que_envejece.py --solo-control
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from comun import auc, cargar_meses, columnas_features, cortes_barrido, etiqueta_binaria, ganancia_por_corte, leer_bo
from config import (DATA_DIR, DATOS, IPC, MESES_ENTRENAMIENTO_FINAL, MESES_ENVEJECE, METODOS_CORRECCION,
                    NOMBRE_FAMILIA, OUTPUT_BARRIDA, RESULTADOS_BARRIDA, RESULTADOS_DIR, SEMILLAS_BARRIDA, ruta_fe)

r6 = importlib.import_module("06_resumen")
r11 = importlib.import_module("11_estres_inflacion")  # modelos finales guardados en data/modelos/
decimal, millones, tabla_md = r6.decimal, r6.millones, r6.tabla_md
FONDO, TINTA, TINTA_2, COLOR = r6.FONDO, r6.TINTA, r6.TINTA_2, r6.COLOR

# Predicción de la prueba de estrés (S7): desde qué inflación acumulada cada
# corrección le gana al modelo sin corregir. La del IPC (+183%) queda fuera del
# rango de esta barrida (llega a +117%).
UMBRAL_PREDICHO = {"mediana": 0.26, "ranking": 0.99}
IPC_ENTRENAMIENTO = np.mean([IPC[m] for m in MESES_ENTRENAMIENTO_FINAL])
CORRECCIONES = [m for m in METODOS_CORRECCION if m != "ninguna"]


def inflacion(mes: int) -> float:
    return IPC[mes] / IPC_ENTRENAMIENTO - 1


def control_datos() -> None:
    """Los meses de 2019 de los datos extendidos son los del experimento principal."""
    t0 = time.time()
    for metodo in METODOS_CORRECCION:
        for mes in range(201903, 201911):
            nuevo = pq.read_table(ruta_fe(metodo, mes))
            viejo = pq.read_table(DATA_DIR / "fe" / metodo / f"{mes}.parquet")
            if nuevo.column_names != viejo.column_names or nuevo.num_rows != viejo.num_rows:
                raise SystemExit(f"{metodo} {mes}: columnas o filas distintas a data/fe")
            for c in nuevo.column_names:
                a, b = nuevo.column(c).to_numpy(), viejo.column(c).to_numpy()
                igual = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
                if not igual:
                    raise SystemExit(f"{metodo} {mes}: la columna {c} no coincide con data/fe")
    print(f"control: los meses 201903-201910 de los datos extendidos son iguales a los del experimento principal "
          f"({time.time() - t0:.0f} s)", flush=True)


def medir() -> pd.DataFrame:
    cortes = cortes_barrido()
    filas, curvas = [], []
    for metodo in METODOS_CORRECCION:
        t0 = time.time()
        corte = int(leer_bo(metodo)["corte_validacion"])
        feats = columnas_features(metodo, MESES_ENVEJECE[0])
        faltan = [s for s in SEMILLAS_BARRIDA if not r11.ruta_modelo(metodo, s).exists()]
        X_train, k_train = cargar_meses(metodo, MESES_ENTRENAMIENTO_FINAL, feats) if faltan else (None, None)
        modelos = r11.modelos_finales(metodo, SEMILLAS_BARRIDA, X_train, k_train, feats)
        del X_train
        for mes in MESES_ENVEJECE:
            X, k = cargar_meses(metodo, [mes], feats)
            clase = k["clase_ternaria"].to_numpy()
            y = etiqueta_binaria(k["clase_ternaria"])
            G = []
            for s, modelo in modelos.items():
                prob = modelo.predict(X)
                g = ganancia_por_corte(prob, clase, cortes)
                G.append(g)
                filas.append({"metodo": metodo, "semilla": s, "mes": mes, "inflacion": inflacion(mes),
                              "corte": corte, "ganancia": int(g[cortes == corte][0]),
                              "ganancia_4000": int(g[cortes == 4000][0]), "ganancia_maxima": int(g.max()),
                              "corte_ganancia_maxima": int(cortes[np.argmax(g)]), "auc": auc(y, prob),
                              "clientes": len(clase), "baja2": int((clase == "BAJA+2").sum())})
            curvas.append(pd.DataFrame({"metodo": metodo, "mes": mes, "corte": cortes,
                                        "ganancia_media": np.mean(G, axis=0).round()}))
            del X
        medias = pd.DataFrame(filas).query("metodo == @metodo").groupby("mes").ganancia.mean()
        print(f"  {NOMBRE_FAMILIA[metodo]}: {millones(medias.iloc[0])} en {MESES_ENVEJECE[0]}, "
              f"{millones(medias.iloc[-1])} en {MESES_ENVEJECE[-1]} ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(filas)

    # control: en 201910 los modelos son los de 04
    for metodo in METODOS_CORRECCION:
        for s in SEMILLAS_BARRIDA:
            final = json.loads((RESULTADOS_DIR / f"final__{metodo}__{s}.json").read_text(encoding="utf-8"))
            g = res.query("metodo == @metodo and semilla == @s and mes == 201910").ganancia.iloc[0]
            if g != final["ganancia_corte_validacion"]:
                raise SystemExit(f"{metodo} semilla {s}: en 201910 da {g:,} y en 04 dio "
                                 f"{final['ganancia_corte_validacion']:,}")
    print("control: en 201910 se recupera la ganancia de 04 en las 5 semillas de las 4 familias", flush=True)

    RESULTADOS_BARRIDA.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTADOS_BARRIDA / "envejece.csv", index=False)
    pd.concat(curvas, ignore_index=True).to_csv(RESULTADOS_BARRIDA / "envejece_curvas.csv", index=False)
    return res


def diferencias(res: pd.DataFrame, metrica: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Por mes: diferencia media de cada corrección contra F0 y en cuántas semillas gana (pareado por semilla)."""
    tabla = res.pivot_table(index=["mes", "semilla"], columns="metodo", values=metrica)
    dif = tabla[CORRECCIONES].sub(tabla["ninguna"], axis=0)
    return dif.groupby("mes").mean(), (dif > 0).groupby("mes").sum()


def resumen(res: pd.DataFrame) -> None:
    OUTPUT_BARRIDA.mkdir(parents=True, exist_ok=True)
    meses = sorted(res.mes.unique())
    infl = res.groupby("mes").inflacion.first()
    media_f0 = res.query("metodo == 'ninguna'").groupby("mes").ganancia.mean()
    baja2 = res.groupby("mes").baja2.first()
    dif, gana = diferencias(res, "ganancia")
    n = len(SEMILLAS_BARRIDA)

    partes = ["# El modelo de 2019 que envejece (generado por scripts/14_modelo_que_envejece.py)\n",
              f"Seguimiento S9 (regla en el README). Modelos finales de cada familia entrenados con "
              f"{MESES_ENTRENAMIENTO_FINAL[0]}-{MESES_ENTRENAMIENTO_FINAL[-1]} (los de 04), aplicados sin reentrenar "
              f"a cada mes. {n} semillas; cada familia con el corte de su bayesiana. \"Inflación acumulada\": IPC del "
              "mes contra el IPC promedio de los meses del entrenamiento. En cada celda, la diferencia media contra "
              f"F0 y en cuántas de las {n} semillas gana.\n"]

    # regla
    filas = []
    for metodo, umbral in UMBRAL_PREDICHO.items():
        arriba = [m for m in meses if infl[m] > umbral]
        ganados = int((dif.loc[arriba, metodo] > 0).sum())
        cumple = len(arriba) > 0 and ganados >= 2 / 3 * len(arriba)
        filas.append({"corrección": NOMBRE_FAMILIA[metodo], "predicción": f"le gana a F0 desde +{umbral:.0%}",
                      "meses por encima": len(arriba), "meses en que le gana": ganados,
                      "diferencia media en esos meses": millones(dif.loc[arriba, metodo].mean()) if arriba else "",
                      "¿se cumple?": "sí" if cumple else "no"})
    ganados_ipc = dif["ipc"] > 0
    desde = next(m for i, m in enumerate(meses) if ganados_ipc.loc[meses[i:]].all())  # gana todos los meses desde acá
    partes += ["## La predicción de la prueba de estrés (regla de S9)\n",
               "Se cumple si la corrección le gana a F0 en al menos 2 de cada 3 meses con más inflación acumulada "
               "que su umbral.\n", tabla_md(pd.DataFrame(filas)), "",
               f"El IPC, que según la prueba de estrés recién convendría desde +183%, no debería ganarle en este "
               f"rango: le gana en {int(ganados_ipc.sum())} de {len(meses)} meses, y en todos desde {desde} "
               f"(+{infl[desde]:.0%}). Tampoco se cumple.\n"]

    # ¿la ventaja crece con la inflación? (descriptivo)
    filas = []
    for metodo in CORRECCIONES:
        rho, p = spearmanr([infl[m] for m in meses], dif[metodo].to_numpy())
        filas.append({"corrección": NOMBRE_FAMILIA[metodo], "meses en que le gana a F0": f"{int((dif[metodo] > 0).sum())} de {len(meses)}",
                      "diferencia media (todos los meses)": millones(dif[metodo].mean()),
                      "correlación con la inflación acumulada (Spearman)": decimal(rho, 2), "p": r6.formato_p(p)})
    partes += ["## ¿La ventaja de corregir crece con la inflación? (descriptivo)\n", tabla_md(pd.DataFrame(filas)), ""]

    # mes a mes
    filas = []
    for m in meses:
        fila = {"mes": m, "inflación acumulada": f"+{infl[m]:.0%}", "BAJA+2 del mes": r6.miles(int(baja2[m])),
                "F0 (sin corregir)": millones(media_f0[m])}
        for metodo in CORRECCIONES:
            fila[NOMBRE_FAMILIA[metodo] + " - F0"] = f"{millones(dif.loc[m, metodo])} ({int(gana.loc[m, metodo])}/{n})"
        filas.append(fila)
    partes += ["## Mes a mes\n", tabla_md(pd.DataFrame(filas)), ""]

    # exploratorio: sin el ruido del corte
    filas = []
    for metrica, nombre in [("ganancia_4000", "todas con 4.000 clientes"),
                            ("ganancia_maxima", "mejor corte de cada semilla (mira el test)")]:
        d, _ = diferencias(res, metrica)
        for metodo in UMBRAL_PREDICHO:
            arriba = [m for m in meses if infl[m] > UMBRAL_PREDICHO[metodo]]
            filas.append({"corte": nombre, "corrección": NOMBRE_FAMILIA[metodo],
                          "meses por encima de su umbral en que le gana a F0": f"{int((d.loc[arriba, metodo] > 0).sum())} de {len(arriba)}",
                          "diferencia media en esos meses": millones(d.loc[arriba, metodo].mean())})
    partes += ["## Exploratorio: con otros cortes (no es parte de la regla)\n", tabla_md(pd.DataFrame(filas)), ""]

    partes += sin_el_corte(res, meses, infl)
    texto = "\n".join(partes)
    (OUTPUT_BARRIDA / "envejece.md").write_text(texto, encoding="utf-8")
    grafico(dif, diferencias(res, "auc")[0] * 1000, infl)
    print(texto)
    print("-> output/barrida/envejece.md y .png, resultados/barrida/envejece.csv")


TRAMOS = [("hasta +30%", 0.0, 0.31), ("+33% a +74%", 0.31, 0.75), ("+81% a +117%", 0.75, 9.0)]


def sin_el_corte(res: pd.DataFrame, meses: list[int], infl: pd.Series) -> list[str]:
    """Exploratorio (agregado al ver los resultados, 19/11): la regla usa el corte fijo de la bayesiana
    de cada familia, pero el mejor corte se mueve mucho entre meses. Acá se comparan las familias con
    medidas que no dependen de ese corte."""
    n = len(SEMILLAS_BARRIDA)
    mejor_f0 = res.query("metodo == 'ninguna'").groupby("mes").corte_ganancia_maxima.mean()
    cortes = {m: int(leer_bo(m)["corte_validacion"]) for m in METODOS_CORRECCION}
    partes = ["## Por qué el corte de la bayesiana confunde (exploratorio, no es parte de la regla)\n",
              f"Cada familia estimula siempre la misma cantidad de clientes: F0 {r6.miles(cortes['ninguna'])}, "
              f"IPC {r6.miles(cortes['ipc'])}, mediana {r6.miles(cortes['mediana'])} y ranking "
              f"{r6.miles(cortes['ranking'])}. Pero el mejor corte de F0 (media de las {n} semillas) pasa de "
              f"{r6.miles(int(round(mejor_f0[meses[0]], -2)))} clientes en {meses[0]} a "
              f"{r6.miles(int(round(mejor_f0.loc[[m for m in meses if m >= 202101]].mean(), -2)))} en promedio en 2021, "
              "porque hay casi el doble de BAJA+2. Con un corte fijo, la familia que estimula más clientes "
              "gana sola en 2021 (el IPC) y la que estimula menos pierde sola (el ranking), corrijan o no. "
              "Por eso se miran también el AUC, que no depende del corte, y la ganancia con el mejor corte de "
              "cada semilla (que mira el test, igual para todas).\n"]
    filas = []
    for metrica, nombre, escala in [("auc", "AUC (milésimas)", 1000), ("ganancia_maxima", "ganancia con el mejor corte", None),
                                    ("ganancia_4000", "ganancia con 4.000 clientes", None),
                                    ("ganancia", "ganancia con el corte de la bayesiana (la regla)", None)]:
        d, _ = diferencias(res, metrica)
        for metodo in CORRECCIONES:
            fila = {"corrección": NOMBRE_FAMILIA[metodo], "medida": nombre,
                    "meses en que le gana a F0": f"{int((d[metodo] > 0).sum())} de {len(meses)}"}
            for tramo, desde, hasta in TRAMOS:
                sel = [m for m in meses if desde <= infl[m] < hasta]
                v = d.loc[sel, metodo]
                media = decimal(v.mean() * escala, 1) if escala else millones(v.mean())
                fila[f"{tramo} ({sel[0]}-{sel[-1]})"] = f"{media} ({int((v > 0).sum())} de {len(sel)})"
            rho, p = spearmanr([infl[m] for m in meses], d[metodo].to_numpy())
            p_texto = r6.formato_p(p)
            fila["Spearman con la inflación"] = f"{decimal(rho, 2)} (p {p_texto if p_texto.startswith('<') else '= ' + p_texto})"
            filas.append(fila)
    partes += ["## Sin depender del corte (exploratorio)\n",
               "Diferencia media contra F0 en cada tramo de inflación acumulada y en cuántos meses del tramo le gana.\n",
               tabla_md(pd.DataFrame(filas)), ""]
    dif_auc, gana_auc = diferencias(res, "auc")
    filas = []
    for m in meses:
        fila = {"mes": m, "inflación acumulada": f"+{infl[m]:.0%}", "mejor corte de F0": r6.miles(int(round(mejor_f0[m], -2)))}
        for metodo in CORRECCIONES:
            fila[NOMBRE_FAMILIA[metodo] + " - F0"] = f"{decimal(dif_auc.loc[m, metodo] * 1000, 1)} ({int(gana_auc.loc[m, metodo])}/{n})"
        filas.append(fila)
    partes += ["## AUC mes a mes (milésimas de AUC de la corrección menos las de F0)\n", tabla_md(pd.DataFrame(filas)), ""]
    return partes


def grafico(dif: pd.DataFrame, dif_auc: pd.DataFrame, infl: pd.Series) -> None:
    meses = list(dif.index)
    x = np.array([infl[m] * 100 for m in meses])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.6), sharex=True)
    paneles = [(ax1, dif / 1e6, "Ganancia con el corte de la bayesiana de cada familia (la regla de S9)",
                "millones de $"),
               (ax2, dif_auc, "AUC, que no depende del corte (exploratorio)", "milésimas de AUC")]
    for ax, valores, titulo, unidad in paneles:
        ax.axhline(0, color=TINTA_2, lw=1.2, ls="--")
        for metodo, umbral in UMBRAL_PREDICHO.items():
            ax.axvline(umbral * 100, color=COLOR[metodo], lw=1.4, ls=":", zorder=1)
        for metodo in CORRECCIONES:
            ax.plot(x, valores[metodo], "-o", color=COLOR[metodo], lw=2, ms=6.5, mec=FONDO, mew=1.5,
                    label=NOMBRE_FAMILIA[metodo], zorder=3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+g}".replace(".", ",").replace("-", "−")
                                                       if v else "0"))
        ax.set_ylabel(f"Corrección menos F0\n({unidad})")
        ax.set_title(titulo, loc="left", fontsize=12, pad=24 if ax is ax1 else 8)  # arriba de ax1 van los meses
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    ax2.set_xlabel("Inflación acumulada entre el entrenamiento (201903-201908) y el mes al que se aplica el modelo")
    fig.suptitle("¿Envejece mejor un modelo corregido?", x=0.06, ha="left", y=0.985, fontsize=16, fontweight="bold")
    fig.text(0.06, 0.95, f"Modelos de 2019 aplicados sin reentrenar a cada mes, de {meses[0]} a {meses[-1]} (media de "
             f"{len(SEMILLAS_BARRIDA)} semillas). Arriba de 0,\nla corrección le gana al modelo sin corregir (F0). "
             "Las punteadas marcan desde dónde ganaría según la prueba\nde estrés: la mediana desde +26% y el "
             "ranking desde +99% (el IPC, desde +183%, queda fuera del gráfico).",
             fontsize=11, color=TINTA_2, va="top", linespacing=1.4)
    # meses de referencia, arriba del primer panel
    arriba = ax1.twiny()
    arriba.set_xlim(ax1.get_xlim())
    marcas = [m for m in meses if m % 100 in (1, 7) or m == meses[0]]
    arriba.set_xticks([infl[m] * 100 for m in marcas], [str(m) for m in marcas], fontsize=10, color=TINTA_2)
    arriba.tick_params(length=0, pad=2)
    for lado in arriba.spines.values():
        lado.set_visible(False)
    arriba.grid(False)
    fig.tight_layout(rect=(0, 0.045, 1, 0.9), h_pad=2.5)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.savefig(OUTPUT_BARRIDA / "envejece.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solo-resumen", action="store_true", help="rehace las tablas y el gráfico con lo ya medido")
    parser.add_argument("--solo-control", action="store_true",
                        help="solo verifica que los meses de 2019 de los datos extendidos sean los del experimento principal")
    args = parser.parse_args()
    if DATOS != "extendido":
        raise SystemExit("Correr con DATOS=extendido (los datos de 2020-2021 están en data/extendido/)")
    if args.solo_control:
        control_datos()
        return
    if args.solo_resumen:
        res = pd.read_csv(RESULTADOS_BARRIDA / "envejece.csv")
    else:
        control_datos()
        res = medir()
    resumen(res)


if __name__ == "__main__":
    main()
