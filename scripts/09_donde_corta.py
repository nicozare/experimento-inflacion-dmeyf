"""¿Dónde cortan los árboles los montos en pesos? (análisis, sin regla de decisión)

Pregunta: los montos son la mitad de la importancia del modelo nominal y aun
así corregirlos por inflación no sube la ganancia. Hipótesis: los cortes que
importan están cerca de cero ("vació la caja de ahorro", "dejó de cobrar el
sueldo"), y a esa altura la inflación no hace cruzar el umbral a casi nadie.

Usa el modelo final de F0 con la primera semilla (entrena 201903-201908, el
mismo que predijo 201910 en 04). Para cada corte de un árbol sobre un monto
(nivel o lag de una variable en pesos) mide:
  - dónde cae el umbral entre los clientes que tienen ese monto con el mismo
    signo: 0% = separa a los que tienen de los que no ("separa el cero");
  - qué parte de los clientes quedaría del otro lado del umbral si todos los
    montos subieran por la inflación: +18% (del mes promedio del
    entrenamiento a octubre) y +27% (de marzo a octubre). Usa la distribución
    de todos los clientes, no solo la de los que llegan a ese nodo.
Cada corte pesa según su ganancia en el árbol (split_gain). Los deltas se
informan aparte: la inflación no los escala, les suma.

Además, la prueba directa: predice 201910 con los montos como están y con
los montos llevados a pesos del entrenamiento (cada columna con el IPC de su
mes) y compara a quién elige el modelo entre los primeros 4.000.

Salida:
  output/donde_corta.csv   un renglón por corte sobre una variable en pesos
  output/donde_corta.png
  output/donde_corta.md

Uso:
    uv run python scripts/09_donde_corta.py
"""
from __future__ import annotations

import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from comun import cargar_meses, columnas_features, es_monetaria, etiqueta_binaria, leer_bo
from config import (COSTO_ESTIMULO, GANANCIA_ACIERTO, IPC, MES_TEST, MESES_ENTRENAMIENTO_FINAL, OUTPUT_DIR,
                    PARAMS_FIJOS, RESULTADOS_DIR, SEMILLAS)

METODO = "ninguna"
PREFIJOS = {"lag_1_": "lag", "lag_2_": "lag", "delta_1_": "delta", "delta_2_": "delta"}
CATEGORIAS = ["separa el cero", "montos bajos (el 25% más bajo)", "montos medios", "montos altos (el 25% más alto)"]
FONDO, TINTA, TINTA_2, GRILLA, AZUL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
IPC_ENTRENAMIENTO = float(np.mean([IPC[m] for m in MESES_ENTRENAMIENTO_FINAL]))
INFLACION = {"18": IPC[MES_TEST] / IPC_ENTRENAMIENTO, "27": IPC[MES_TEST] / IPC[MESES_ENTRENAMIENTO_FINAL[0]]}

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.8, "legend.frameon": False,
})


def separar(feature: str) -> tuple[str, str]:
    for prefijo, tipo in PREFIJOS.items():
        if feature.startswith(prefijo):
            return feature[len(prefijo):], tipo
    return feature, "nivel"


def medir_cortes(umbrales: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    """Para cada umbral sobre la columna x: percentil entre los clientes con
    monto del mismo signo, y fracción de clientes que cruzaría el umbral si
    los montos subieran por cada inflación de INFLACION."""
    xs = np.sort(x[~np.isnan(x)])
    n = len(xs)
    n_neg, inicio_pos = np.searchsorted(xs, 0, "left"), np.searchsorted(xs, 0, "right")
    n_pos = n - inicio_pos
    debajo = np.searchsorted(xs, umbrales, "right")
    pct = np.where(umbrales >= 0,
                   (debajo - inicio_pos) / max(n_pos, 1),   # positivos entre 0 y el umbral
                   (n_neg - debajo) / max(n_neg, 1))        # negativos entre el umbral y 0
    salida = {"percentil": np.clip(pct, 0, 1)}
    for nombre, f in INFLACION.items():
        bajo, alto = np.minimum(umbrales / f, umbrales), np.maximum(umbrales / f, umbrales)
        salida[f"cruzan_{nombre}"] = (np.searchsorted(xs, alto, "right") - np.searchsorted(xs, bajo, "right")) / n
    return pd.DataFrame(salida)


def categoria(p: float) -> str:
    if p == 0:
        return CATEGORIAS[0]
    return CATEGORIAS[1] if p <= 0.25 else CATEGORIAS[2] if p <= 0.75 else CATEGORIAS[3]


def mes_anterior(mes: int, k: int) -> int:
    total = (mes // 100) * 12 + mes % 100 - 1 - k
    return (total // 12) * 100 + total % 12 + 1


def deflactar(X: np.ndarray, feats: list[str], deltas: bool, mes: int = MES_TEST) -> np.ndarray:
    """El mes (por defecto 201910) en pesos del entrenamiento: cada monto con
    el IPC de su mes (nivel `mes`, lag_1 el mes anterior, lag_2 dos meses
    antes). Con deltas=True los deltas se recalculan con los montos
    deflactados; con False quedan nominales (aísla el efecto de los umbrales
    sobre montos). 14 lo usa con otros meses (control de S9)."""
    D = X.copy()
    idx = {f: j for j, f in enumerate(feats)}
    meses = {"": mes, "lag_1_": mes_anterior(mes, 1), "lag_2_": mes_anterior(mes, 2)}
    for base in [f for f in feats if separar(f)[1] == "nivel" and es_monetaria(f)]:
        for prefijo, mes in meses.items():
            D[:, idx[prefijo + base]] *= IPC_ENTRENAMIENTO / IPC[mes]
        if deltas:
            D[:, idx["delta_1_" + base]] = D[:, idx[base]] - D[:, idx["lag_1_" + base]]
            D[:, idx["delta_2_" + base]] = D[:, idx[base]] - D[:, idx["lag_2_" + base]]
    return D


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def millones(x: float) -> str:
    return f"${x / 1e6:,.1f}M".replace(",", "X").replace(".", ",").replace("X", ".")


def ganancia(prob: np.ndarray, clase: np.ndarray, corte: int) -> int:
    elegidos = np.argsort(-prob, kind="stable")[:corte]
    aciertos = int(np.sum(clase[elegidos] == "BAJA+2"))
    return aciertos * GANANCIA_ACIERTO - (corte - aciertos) * COSTO_ESTIMULO


def main() -> None:
    t0 = time.time()
    bo = leer_bo(METODO)
    semilla, corte = SEMILLAS[0], int(bo["corte_validacion"])
    feats = columnas_features(METODO, MES_TEST)
    X, k = cargar_meses(METODO, MESES_ENTRENAMIENTO_FINAL, feats)
    # igual que 04: el dataset se arma ANTES de entrenar y sin la semilla, si no
    # LightGBM elige los bins con la semilla del modelo y sale otro modelo
    dset = lgb.Dataset(X, label=etiqueta_binaria(k["clase_ternaria"]), feature_name=feats, free_raw_data=False,
                       params={c: PARAMS_FIJOS[c] for c in ("feature_pre_filter", "verbosity", "num_threads")}
                       ).construct()
    modelo = lgb.train({**PARAMS_FIJOS, **bo["hiperparametros"], "seed": semilla}, dset,
                       num_boost_round=int(bo["best_iteration"]))
    print(f"modelo F0 semilla {semilla}: {bo['best_iteration']} árboles ({time.time() - t0:.0f} s)", flush=True)

    # --- 1. los cortes de los árboles
    arboles = modelo.trees_to_dataframe()
    cortes = arboles[arboles.split_feature.notna()][["tree_index", "split_feature", "split_gain", "threshold"]]
    ganancia_total = cortes.split_gain.sum()
    cortes = cortes.assign(base=cortes.split_feature.map(lambda f: separar(f)[0]),
                           tipo=cortes.split_feature.map(lambda f: separar(f)[1]))
    cortes = cortes[cortes.base.map(es_monetaria)].reset_index(drop=True)
    col = {f: j for j, f in enumerate(feats)}
    medidas = []
    for feature, grupo in cortes.groupby("split_feature"):
        m = medir_cortes(grupo.threshold.to_numpy(float), X[:, col[feature]].astype(float))
        medidas.append(m.set_index(grupo.index))
    cortes = cortes.join(pd.concat(medidas))
    cortes["categoria"] = cortes.percentil.map(categoria)
    cortes.to_csv(OUTPUT_DIR / "donde_corta.csv", index=False)

    montos = cortes[cortes.tipo != "delta"]
    peso = montos.split_gain / montos.split_gain.sum()
    por_categoria = (montos.groupby("categoria").split_gain.sum() / montos.split_gain.sum() * 100).reindex(
        CATEGORIAS).fillna(0)
    cruzan = {n: float(np.sum(peso * montos[f"cruzan_{n}"]) * 100) for n in INFLACION}
    pct_montos = montos.split_gain.sum() / ganancia_total * 100
    pct_deltas = cortes[cortes.tipo == "delta"].split_gain.sum() / ganancia_total * 100

    # --- 2. prueba directa: octubre en pesos del entrenamiento
    X_test, k_test = cargar_meses(METODO, [MES_TEST], feats)
    clase = k_test["clase_ternaria"].to_numpy()
    p_nominal = modelo.predict(X_test)
    top_nom = set(np.argsort(-p_nominal, kind="stable")[:corte])
    g_nom = ganancia(p_nominal, clase, corte)
    variantes = {}
    for nombre, deltas in (("montos", False), ("todo", True)):
        p = modelo.predict(deflactar(X_test, feats, deltas))
        variantes[nombre] = {
            "coinciden": len(top_nom & set(np.argsort(-p, kind="stable")[:corte])) / corte * 100,
            "rho": spearmanr(p_nominal, p).statistic, "ganancia": ganancia(p, clase, corte)}
    guardada = RESULTADOS_DIR / f"prediccion__{METODO}__{semilla}.parquet"
    if guardada.exists():  # mismo modelo que el de 04: tiene que dar las mismas probabilidades
        igual = np.allclose(pd.read_parquet(guardada).prob.to_numpy(), p_nominal)
        print(f"control: predicción idéntica a la de 04 -> {'sí' if igual else 'NO'}", flush=True)

    # --- salida
    fmt = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")  # noqa: E731
    lineas = [
        "# ¿Dónde cortan los árboles los montos en pesos? (generado por scripts/09_donde_corta.py)\n",
        f"Modelo final de F0 (semilla {semilla}, entrena {MESES_ENTRENAMIENTO_FINAL[0]}-"
        f"{MESES_ENTRENAMIENTO_FINAL[-1]}). Cada corte pesa según su ganancia en el árbol.\n",
        f"- Los cortes sobre montos (nivel y lags de las variables en pesos) son el {fmt(pct_montos)}% de la "
        f"ganancia de todos los cortes; los deltas de esas variables, otro {fmt(pct_deltas)}%.",
        "- Dónde caen los cortes sobre montos (% de su ganancia), según el lugar del umbral entre los clientes "
        "que tienen ese monto:\n",
        "| dónde corta | % de la ganancia |", "|---|---|",
        *[f"| {c} | {fmt(v)}% |" for c, v in por_categoria.items()],
        "",
        f"- Si todos los montos subieran un {fmt((INFLACION['18'] - 1) * 100, 0)}% (del mes promedio del "
        f"entrenamiento a octubre), cada corte cambiaría de rama, en promedio, al {fmt(cruzan['18'])}% de los "
        f"clientes; con un {fmt((INFLACION['27'] - 1) * 100, 0)}% (de marzo a octubre), al {fmt(cruzan['27'])}%.",
        f"- Prueba directa en {MES_TEST}: se le da al mismo modelo octubre llevado a pesos del entrenamiento "
        f"(cada monto con el IPC de su mes) y se compara con octubre nominal, en los primeros {miles(corte)} "
        "(la ganancia mira el test: solo descriptivo).\n",
        f"| octubre | mismos clientes en los primeros {miles(corte)} | correlación de rangos | ganancia |",
        "|---|---|---|---|",
        f"| nominal (como lo usa F0) | 100% | 1,000 | {millones(g_nom)} |",
        f"| montos deflactados, deltas nominales | {fmt(variantes['montos']['coinciden'])}% | "
        f"{fmt(variantes['montos']['rho'], 3)} | {millones(variantes['montos']['ganancia'])} |",
        f"| montos y deltas deflactados | {fmt(variantes['todo']['coinciden'])}% | "
        f"{fmt(variantes['todo']['rho'], 3)} | {millones(variantes['todo']['ganancia'])} |",
    ]
    (OUTPUT_DIR / "donde_corta.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(CATEGORIAS))
    ax.set_axisbelow(True)
    ax.barh(y, por_categoria.to_numpy(), height=0.55, color=AZUL)
    for yi, v in zip(y, por_categoria.to_numpy()):
        ax.annotate(f"{fmt(v)}%", (v, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=12)
    ax.set_yticks(y, CATEGORIAS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, por_categoria.max() * 1.18)
    ax.set_xlabel("% de la ganancia de los cortes sobre montos en pesos (nivel y lags)")
    ax.set_title("¿Dónde cortan los árboles los montos en pesos?", pad=50)
    ax.text(0, 1.025, f"Modelo final de F0 (semilla {semilla}); cada corte pesa según su ganancia.\n"
            f"Si los montos subieran {fmt((INFLACION['18'] - 1) * 100, 0)}%, cada corte cambiaría de rama, en "
            f"promedio, al {fmt(cruzan['18'])}% de los clientes.", transform=ax.transAxes, fontsize=11,
            color=TINTA_2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "donde_corta.png", dpi=160)
    plt.close(fig)
    print(f"\n-> output/donde_corta.md, .csv y .png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
