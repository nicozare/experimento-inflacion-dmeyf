"""S10. La barrida con ventana móvil (barrida de meses, 19/11; regla en el README).

Para cada mes de test t de config.MESES_TEST_BARRIDA, la ventana del baseline
corrida en el tiempo: entrena t-7 a t-4, elige el corte en t-2 (corte estable:
el máximo de la curva promedio de las semillas, como S1), reentrena t-7 a t-2
y mide la ganancia en t. Hiperparámetros y cantidad de árboles: los de la
bayesiana de 2019 de cada familia (no se rehace la bayesiana en cada mes).
5 semillas (config.SEMILLAS_BARRIDA).

Familias: F0 a F3 y las tres de las hipótesis secundarias del tutor (F6, F7 y
F8; ver config.FAMILIAS_DRIFT). Sus variables se eligen una vez por ventana,
con el PSI de 05 entre los meses del reentreno y el mes de test (sin la clase),
y se usan igual en los modelos de validación y en los finales.

Con t = 201910, F0 a F3 son exactamente los modelos de 07 y 04: se reusan sus
curvas (las 5 primeras semillas). Control: F0 con la semilla 100081 se vuelve a
entrenar y tiene que dar las mismas curvas.

Retoma solo: cada (mes, familia, semilla) guarda sus curvas y se saltea si ya
están. Primero corre F0 a F3 en todos los meses y después las familias nuevas.

Salida:
  resultados/barrida/movil/<mes>/curva_validacion__<familia>__<semilla>.csv, curva__... y sus .json
  resultados/barrida/movil/<mes>/variables_drift.csv   PSI de cada variable y qué se saca o se rankea
  resultados/barrida/movil.csv                         ganancia por mes, familia y semilla en su corte estable
  output/barrida/movil.md y .png

Uso:
    DATOS=extendido uv run python scripts/15_barrida_movil.py
    DATOS=extendido uv run python scripts/15_barrida_movil.py --familias ninguna ipc mediana ranking
    DATOS=extendido uv run python scripts/15_barrida_movil.py --solo-resumen
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from comun import auc, cargar_meses, columnas_features, cortes_barrido, es_monetaria, etiqueta_binaria, \
    ganancia_por_corte, leer_bo
from config import (DATOS, FAMILIAS_DRIFT, MESES_TEST_BARRIDA, METODOS_CORRECCION, NOMBRE_FAMILIA, OUTPUT_BARRIDA,
                    OUTPUT_DATOS, PARAMS_FIJOS, PSI_CON_DRIFT, PSI_MUCHO_DRIFT, RESULTADOS_BARRIDA, RESULTADOS_DIR,
                    SEMILLAS_BARRIDA)

r5 = importlib.import_module("05_drift")      # psi()
r6 = importlib.import_module("06_resumen")    # formatos y colores
r9 = importlib.import_module("09_donde_corta")  # separar(): nombre base de cada columna
decimal, millones, tabla_md, formato_p = r6.decimal, r6.millones, r6.tabla_md, r6.formato_p
FONDO, TINTA, TINTA_2, COLOR = r6.FONDO, r6.TINTA, r6.TINTA_2, r6.COLOR

FAMILIAS = METODOS_CORRECCION + list(FAMILIAS_DRIFT)
CARPETA = RESULTADOS_BARRIDA / "movil"
CORTES = cortes_barrido()
SEMILLA_CONTROL = SEMILLAS_BARRIDA[0]
# comparaciones de la regla: todas contra F0, y cada familia nueva contra la suya
COMPARACIONES = [(f, "ninguna") for f in FAMILIAS if f != "ninguna"] + \
                [("mediana_sin_drift", "mediana"), ("ranking_con_drift", "ranking")]


def sumar_meses(mes: int, k: int) -> int:
    total = (mes // 100) * 12 + mes % 100 - 1 + k
    return (total // 12) * 100 + total % 12 + 1


def ventana(t: int) -> dict:
    return {"entrenamiento": [sumar_meses(t, k) for k in range(-7, -3)], "validacion": sumar_meses(t, -2),
            "final": [sumar_meses(t, k) for k in range(-7, -1)], "test": t}


# ---------------------------------------------------------------------------
# Variables de cada familia en cada ventana
# ---------------------------------------------------------------------------
def variables_drift(t: int, feats: list[str]) -> pd.DataFrame:
    """PSI de cada variable (nivel) entre los meses del reentreno y el de test,
    sin corregir y después de dividir por la mediana. Se calcula una vez por ventana."""
    ruta = CARPETA / str(t) / "variables_drift.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    v = ventana(t)
    base = [f for f in feats if r9.separar(f)[1] == "nivel"]
    filas = {c: {"variable": c, "en_pesos": es_monetaria(c)} for c in base}
    for metodo, col in [("ninguna", "psi_sin_corregir"), ("mediana", "psi_mediana")]:
        X_ref, _ = cargar_meses(metodo, v["final"], base)
        X_t, _ = cargar_meses(metodo, [t], base)
        for j, c in enumerate(base):
            filas[c][col] = r5.psi(X_ref[:, j].astype(float), X_t[:, j].astype(float))
        del X_ref, X_t
    d = pd.DataFrame(list(filas.values()))
    d["sale_de_F6"] = d.psi_sin_corregir > PSI_MUCHO_DRIFT
    d["sale_de_F7"] = d.psi_mediana > PSI_MUCHO_DRIFT
    d["rankeada_en_F8"] = ~d.en_pesos & (d.psi_sin_corregir > PSI_CON_DRIFT)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(ruta, index=False)
    return d


def columnas(familia: str, t: int, feats: dict[str, list[str]]) -> tuple[str, list[str], list[str]]:
    """(datos de los que sale la familia, sus columnas, columnas que se toman de ranking_todo)."""
    if familia in METODOS_CORRECCION:
        return familia, feats[familia], []
    d = variables_drift(t, feats["ninguna"]).set_index("variable")
    if familia == "ninguna_sin_drift":
        fuera = set(d.index[d.sale_de_F6])
        return "ninguna", [c for c in feats["ninguna"] if r9.separar(c)[0] not in fuera], []
    if familia == "mediana_sin_drift":
        fuera = set(d.index[d.sale_de_F7])
        return "mediana", [c for c in feats["mediana"] if r9.separar(c)[0] not in fuera], []
    if familia == "ranking_con_drift":
        rankear = set(d.index[d.rankeada_en_F8])
        return "ranking", feats["ranking"], [c for c in feats["ranking"] if r9.separar(c)[0] in rankear]
    raise ValueError(familia)


def cargar(datos: str, cols: list[str], de_ranking_todo: list[str], meses: list[int]):
    X, k = cargar_meses(datos, meses, cols)
    if de_ranking_todo:
        Xr, kr = cargar_meses("ranking_todo", meses, de_ranking_todo)
        if not np.array_equal(kr["numero_de_cliente"].to_numpy(), k["numero_de_cliente"].to_numpy()):
            raise SystemExit("ranking_todo no tiene los mismos clientes en el mismo orden")
        posicion = {c: j for j, c in enumerate(cols)}
        X[:, [posicion[c] for c in de_ranking_todo]] = Xr
        del Xr
    return X, k


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(bo: dict, X_tr, k_tr, X_ev, k_ev, cols: list[str], semillas: list[int], carpeta, prefijo: str,
             familia: str) -> None:
    """Igual que 04 y 07: el dataset se arma una vez, antes de entrenar y sin la semilla."""
    params_dataset = {c: PARAMS_FIJOS[c] for c in ("feature_pre_filter", "verbosity", "num_threads")}
    dtrain = lgb.Dataset(X_tr, label=etiqueta_binaria(k_tr["clase_ternaria"]), feature_name=cols,
                         params=params_dataset, free_raw_data=False).construct()
    clase = k_ev["clase_ternaria"].to_numpy()
    y = etiqueta_binaria(k_ev["clase_ternaria"])
    for s in semillas:
        inicio = time.time()
        modelo = lgb.train({**PARAMS_FIJOS, **bo["hiperparametros"], "seed": s}, dtrain,
                           num_boost_round=int(bo["best_iteration"]))
        prob = modelo.predict(X_ev)
        g = ganancia_por_corte(prob, clase, CORTES)
        pd.DataFrame({"corte": CORTES, "ganancia": g}).to_csv(carpeta / f"{prefijo}__{familia}__{s}.csv", index=False)
        (carpeta / f"{prefijo}__{familia}__{s}.json").write_text(json.dumps({
            "familia": familia, "semilla": s, "auc": auc(y, prob), "ganancia_maxima": int(g.max()),
            "columnas": len(cols), "arboles": int(bo["best_iteration"]), "segundos": round(time.time() - inicio, 1),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    {prefijo} semilla {s}: máximo ${g.max() / 1e6:,.1f}M | {time.time() - inicio:.0f} s", flush=True)
    del dtrain


def entrenar_ventana(t: int, familia: str, feats: dict, semillas: list[int], carpeta) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    v = ventana(t)
    bo = leer_bo(FAMILIAS_DRIFT.get(familia, familia))
    datos, cols, de_todo = columnas(familia, t, feats)
    for prefijo, meses_tr, mes_ev in [("curva_validacion", v["entrenamiento"], v["validacion"]),
                                      ("curva", v["final"], t)]:
        pendientes = [s for s in semillas if not (carpeta / f"{prefijo}__{familia}__{s}.csv").exists()]
        if not pendientes:
            continue
        t0 = time.time()
        X_tr, k_tr = cargar(datos, cols, de_todo, meses_tr)
        X_ev, k_ev = cargar(datos, cols, de_todo, [mes_ev])
        print(f"  {t} {NOMBRE_FAMILIA[familia]} {prefijo}: entrena {meses_tr[0]}-{meses_tr[-1]} ({len(X_tr):,} filas, "
              f"{len(cols)} columnas{f', {len(de_todo)} de ranking_todo' if de_todo else ''}), predice {mes_ev} "
              f"(datos en {time.time() - t0:.0f} s)", flush=True)
        entrenar(bo, X_tr, k_tr, X_ev, k_ev, cols, pendientes, carpeta, prefijo, familia)
        del X_tr, X_ev


def correr(t: int, familia: str, feats: dict) -> None:
    carpeta = CARPETA / str(t)
    if t == 201910 and familia in METODOS_CORRECCION:
        # la ventana principal: los modelos son los de 07 (validación) y 04 (final)
        carpeta.mkdir(parents=True, exist_ok=True)
        for s in SEMILLAS_BARRIDA:
            for prefijo in ("curva_validacion", "curva"):
                shutil.copyfile(RESULTADOS_DIR / f"{prefijo}__{familia}__{s}.csv", carpeta / f"{prefijo}__{familia}__{s}.csv")
        return
    entrenar_ventana(t, familia, feats, SEMILLAS_BARRIDA, carpeta)


def control_201910(feats: dict) -> None:
    carpeta = CARPETA / "control_201910"
    entrenar_ventana(201910, "ninguna", feats, [SEMILLA_CONTROL], carpeta)
    for prefijo in ("curva_validacion", "curva"):
        nueva = pd.read_csv(carpeta / f"{prefijo}__ninguna__{SEMILLA_CONTROL}.csv")
        vieja = pd.read_csv(RESULTADOS_DIR / f"{prefijo}__ninguna__{SEMILLA_CONTROL}.csv")
        if not nueva.equals(vieja):
            raise SystemExit(f"control: {prefijo} de F0 en 201910 no coincide con la del experimento principal")
    print("control: en 201910, F0 con la semilla 100081 reproduce las curvas de 07 y 04", flush=True)


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def leer_curvas(t: int, familia: str, prefijo: str) -> pd.DataFrame | None:
    rutas = [CARPETA / str(t) / f"{prefijo}__{familia}__{s}.csv" for s in SEMILLAS_BARRIDA]
    if not all(r.exists() for r in rutas):
        return None
    return pd.DataFrame({s: pd.read_csv(r).set_index("corte").ganancia for s, r in zip(SEMILLAS_BARRIDA, rutas)}).T


def resumen() -> None:
    filas = []
    for t in MESES_TEST_BARRIDA:
        for familia in FAMILIAS:
            V, T = leer_curvas(t, familia, "curva_validacion"), leer_curvas(t, familia, "curva")
            if V is None or T is None:
                continue
            corte = int(V.mean().idxmax())  # corte estable, como S1
            for s in SEMILLAS_BARRIDA:
                filas.append({"mes": t, "familia": familia, "semilla": s, "corte_estable": corte,
                              "ganancia": int(T.loc[s, corte]), "ganancia_maxima": int(T.loc[s].max())})
    res = pd.DataFrame(filas)
    RESULTADOS_BARRIDA.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTADOS_BARRIDA / "movil.csv", index=False)
    OUTPUT_BARRIDA.mkdir(parents=True, exist_ok=True)

    G = res.pivot_table(index="mes", columns="familia", values="ganancia")  # media de las semillas
    S = res.pivot_table(index=["mes", "semilla"], columns="familia", values="ganancia")
    cortes = res.groupby(["mes", "familia"]).corte_estable.first().unstack()
    familias = [f for f in FAMILIAS if f in G.columns]
    n = len(SEMILLAS_BARRIDA)

    # regla
    veredictos = []
    for a, b in COMPARACIONES:
        if a not in G.columns or b not in G.columns:
            continue
        meses = G.index[G[a].notna() & G[b].notna()]
        d = G.loc[meses, a] - G.loc[meses, b]
        p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else np.nan
        veredicto = ("supera" if d.mean() > 0 and p < 0.05 else "pierde" if d.mean() < 0 and p < 0.05 else "empata")
        completa = "" if len(meses) == len(MESES_TEST_BARRIDA) else f" (con {len(meses)} de {len(MESES_TEST_BARRIDA)} meses)"
        veredictos.append({"comparación": f"{NOMBRE_FAMILIA[a]} vs {NOMBRE_FAMILIA[b]}",
                           "diferencia media por mes": millones(d.mean()), "meses en que gana": f"{int((d > 0).sum())} de {len(d)}",
                           "Wilcoxon p (por mes)": formato_p(p), "resultado": veredicto + completa})

    partes = ["# Barrida con ventana móvil (generado por scripts/15_barrida_movil.py)\n",
              f"Seguimiento S10 (regla en el README). Para cada mes de test t: entrena t-7 a t-4, elige el corte "
              f"estable en t-2, reentrena t-7 a t-2 y mide la ganancia en t. Hiperparámetros de la bayesiana de 2019 de "
              f"cada familia, {n} semillas. Una familia supera a otra si la diferencia media por mes es positiva y el "
              "Wilcoxon pareado por mes da p < 0,05; pierde si es negativa con p < 0,05; si no, empata.\n",
              "## Resultado de la regla\n", tabla_md(pd.DataFrame(veredictos)), ""]

    filas = []
    clase = pd.read_csv(OUTPUT_DATOS / "clase_ternaria_por_mes.csv", index_col=0) \
        if (OUTPUT_DATOS / "clase_ternaria_por_mes.csv").exists() else None
    for t in G.index:
        v = ventana(t)
        fila = {"mes de test": t, "entrena / valida": f"{v['final'][0]}-{v['final'][-1]} / {v['validacion']}"}
        if clase is not None and t in clase.index:
            fila["BAJA+2 del mes"] = r6.miles(int(clase.loc[t, "BAJA+2"]))
        fila["F0 (corte)"] = f"{millones(G.loc[t, 'ninguna'])} ({r6.miles(int(cortes.loc[t, 'ninguna']))})" \
            if "ninguna" in G.columns and pd.notna(G.loc[t, "ninguna"]) else ""
        for f in familias:
            if f == "ninguna":
                continue
            if pd.isna(G.loc[t, f]) or pd.isna(G.loc[t, "ninguna"]):
                fila[NOMBRE_FAMILIA[f]] = ""
                continue
            gana = int((S.loc[t, f] > S.loc[t, "ninguna"]).sum())
            fila[NOMBRE_FAMILIA[f]] = f"{millones(G.loc[t, f] - G.loc[t, 'ninguna'])} ({gana}/{n})"
        filas.append(fila)
    partes += ["## Mes a mes: diferencia contra F0 (en cuántas semillas gana)\n", tabla_md(pd.DataFrame(filas)), ""]

    drift = []
    for t in MESES_TEST_BARRIDA:
        ruta = CARPETA / str(t) / "variables_drift.csv"
        if ruta.exists():
            d = pd.read_csv(ruta)
            drift.append({"mes de test": t, "sacadas en F6": int(d.sale_de_F6.sum()), "sacadas en F7": int(d.sale_de_F7.sum()),
                          "rankeadas de más en F8": int(d.rankeada_en_F8.sum()),
                          "las de más drift (PSI sin corregir)": ", ".join(
                              d.sort_values("psi_sin_corregir", ascending=False).variable.head(4))})
    if drift:
        partes += [f"## Variables con drift en cada ventana (PSI > {decimal(PSI_MUCHO_DRIFT, 2)} se sacan; "
                   f"no monetarias con PSI > {decimal(PSI_CON_DRIFT, 1)} se rankean)\n", tabla_md(pd.DataFrame(drift)), ""]

    texto = "\n".join(partes)
    (OUTPUT_BARRIDA / "movil.md").write_text(texto, encoding="utf-8")
    if "ninguna" in G.columns and len(familias) > 1:
        grafico(G, familias)
    print(texto)
    print("-> output/barrida/movil.md y .png, resultados/barrida/movil.csv")


def grafico(G: pd.DataFrame, familias: list[str]) -> None:
    otras = [f for f in familias if f != "ninguna"]
    color = {f: COLOR.get(f, COLOR.get(FAMILIAS_DRIFT.get(f), TINTA_2)) for f in otras}
    alto = 2.8 + 0.85 * len(otras)
    fig, ax = plt.subplots(figsize=(11, alto))
    for i, f in enumerate(otras):
        d = (G[f] - G["ninguna"]).dropna() / 1e6
        hueco = f in FAMILIAS_DRIFT  # las familias nuevas, con el color de la familia de la que salen y sin relleno
        ax.scatter(d, np.full(len(d), i), s=70, color=FONDO if hueco else color[f], edgecolor=color[f],
                   linewidth=2, zorder=3)
        ax.plot([d.mean()] * 2, [i - 0.3, i + 0.3], color=TINTA, lw=2.5, zorder=4)
        ax.annotate(f"media {millones(d.mean() * 1e6)} · gana en {int((d > 0).sum())} de {len(d)} meses",
                    (1.0, i), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=11, color=TINTA_2)
    ax.axvline(0, color=TINTA_2, lw=1.2, ls="--")
    ax.set_yticks(range(len(otras)), [NOMBRE_FAMILIA[f] for f in otras])
    ax.set_ylim(len(otras) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    ax.set_xlabel("Ganancia de la familia menos la de F0 en el mismo mes (millones de $)")
    ax.set_title("Barrida con ventana móvil: cada familia contra F0, mes por mes", pad=50)
    ax.text(0, 1.025, f"Cada punto es un mes de test ({', '.join(map(str, G.index))}); la raya negra es la media.\n"
            f"Media de {len(SEMILLAS_BARRIDA)} semillas, corte estable de cada familia. Huecos: hipótesis del tutor.",
            transform=ax.transAxes, fontsize=11, color=TINTA_2, va="bottom")
    fig.tight_layout()
    fig.subplots_adjust(right=0.66)
    fig.savefig(OUTPUT_BARRIDA / "movil.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--familias", nargs="+", default=FAMILIAS, choices=FAMILIAS)
    parser.add_argument("--meses", nargs="+", type=int, default=MESES_TEST_BARRIDA, choices=MESES_TEST_BARRIDA)
    parser.add_argument("--solo-resumen", action="store_true", help="rehace las tablas y el gráfico con lo ya corrido")
    args = parser.parse_args()
    if DATOS != "extendido":
        raise SystemExit("Correr con DATOS=extendido (los datos de 2020-2021 están en data/extendido/)")
    if not args.solo_resumen:
        feats = {m: columnas_features(m, MESES_TEST_BARRIDA[0]) for m in METODOS_CORRECCION}
        control_201910(feats)  # si ya corrió, solo vuelve a comparar
        for familia in args.familias:
            t0 = time.time()
            for t in args.meses:
                correr(t, familia, feats)
            print(f"== {NOMBRE_FAMILIA[familia]}: lista en los {len(args.meses)} meses ({(time.time() - t0) / 60:.0f} min)",
                  flush=True)
            try:  # el resumen se rehace después de cada familia, para mirar a mitad de camino
                resumen()
            except Exception as e:  # noqa: BLE001 (un resumen intermedio no corta la corrida)
                print(f"  (el resumen intermedio falló: {e!r}; la corrida sigue)", flush=True)
    resumen()


if __name__ == "__main__":
    main()
