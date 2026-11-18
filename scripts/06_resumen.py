"""Junta los resultados de las 4 familias: tablas, tests y gráficos para el video.

Lee resultados/bo__*.json (bayesiana), resultados/final__*.json y
resultados/curva__*.csv (semillerío en 201910) y output/drift_*.csv (05).

Regla de decisión (fijada ANTES de ver el test):
  Una familia supera al baseline (F0) si
    1. su ganancia media en 201910, en el corte elegido en validación, supera
       a la de F0 por más que el desvío entre semillas de F0, y
    2. el test de Wilcoxon pareado por semilla (bilateral) da p < 0,05.
  Para la segunda parte de la hipótesis ("rinde al menos como el IPC"): F2 y
  F3 empatan o le ganan a F1 si su diferencia media contra F1 es >= 0 o si
  el test de Wilcoxon contra F1 no da significativo (p >= 0,05).

Salida: output/resumen.md, output/resultados_por_semilla.csv y los gráficos
output/curvas_ganancia_201910.png, output/diferencia_vs_baseline.png y
output/drift_psi.png.

Uso:
    uv run python scripts/06_resumen.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import COSTO_ESTIMULO, GANANCIA_ACIERTO, MES_TEST, METODOS, NOMBRE_FAMILIA, OUTPUT_DIR, RESULTADOS_DIR

# Paleta categórica de referencia (orden fijo; color = familia en todos los gráficos)
COLOR = {"ninguna": "#2a78d6", "ipc": "#eb6834", "mediana": "#1baf7a", "ranking": "#eda100"}
FONDO, TINTA, TINTA_2, GRILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.8, "legend.frameon": False,
})


def millones(x: float) -> str:
    return f"${x / 1e6:,.1f}M".replace(",", "X").replace(".", ",").replace("X", ".")


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def cargar() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    bo = {m: json.loads((RESULTADOS_DIR / f"bo__{m}.json").read_text(encoding="utf-8"))
          for m in METODOS if (RESULTADOS_DIR / f"bo__{m}.json").exists()}
    finales = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in RESULTADOS_DIR.glob("final__*.json")])
    curvas = []
    for p in RESULTADOS_DIR.glob("curva__*.csv"):
        _, metodo, semilla = p.stem.split("__")
        c = pd.read_csv(p)
        c["metodo"], c["semilla"] = metodo, int(semilla)
        curvas.append(c)
    return bo, finales, (pd.concat(curvas, ignore_index=True) if curvas else pd.DataFrame())


def comparar(finales: pd.DataFrame, metodo: str, contra: str, metrica: str) -> dict:
    a = finales[finales.metodo == metodo].set_index("semilla")[metrica]
    b = finales[finales.metodo == contra].set_index("semilla")[metrica]
    comunes = a.index.intersection(b.index)
    dif = (a.loc[comunes] - b.loc[comunes]).astype(float)
    p = wilcoxon(dif).pvalue if len(comunes) >= 6 and (dif != 0).any() else np.nan
    return {"semillas": len(comunes), "dif_media": dif.mean(), "gana_en": int((dif > 0).sum()), "p_wilcoxon": p,
            "desvio_contra": b.loc[comunes].std()}


def tabla_md(df: pd.DataFrame) -> str:
    filas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    filas += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(filas)


def grafico_curvas(curvas: pd.DataFrame, bo: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for metodo in METODOS:
        c = curvas[curvas.metodo == metodo]
        if c.empty:
            continue
        g = c.groupby("corte")["ganancia"].agg(["mean", "std"]) / 1e6
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[metodo], alpha=0.12, lw=0)
        ax.plot(g.index, g["mean"], color=COLOR[metodo], lw=2.2, label=NOMBRE_FAMILIA[metodo])
        corte = bo.get(metodo, {}).get("corte_validacion")
        if corte in g.index:
            ax.plot([corte], [g.loc[corte, "mean"]], "o", ms=9, color=COLOR[metodo], mec=FONDO, mew=2, zorder=5)
        fin = g.index.max()
        ax.annotate(NOMBRE_FAMILIA[metodo], (fin, g.loc[fin, "mean"]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=12, color=TINTA)
    ax.set_title(f"Ganancia en {MES_TEST} según cuántos clientes se estimulan")
    ax.set_xlabel("Clientes estimulados (los de mayor probabilidad)")
    ax.set_ylabel("Ganancia (millones de $)")
    ax.text(0, 1.01, f"Media de {curvas.semilla.nunique()} semillas por familia; banda = ±1 desvío; "
            "punto = corte elegido en validación (201908)", transform=ax.transAxes, fontsize=11, color=TINTA_2)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.27))
    ax.set_xlim(right=curvas.corte.max() * 1.13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "curvas_ganancia_201910.png", dpi=160)
    plt.close(fig)


def grafico_diferencias(finales: pd.DataFrame) -> None:
    base = finales[finales.metodo == "ninguna"].set_index("semilla")["ganancia_corte_validacion"]
    otros = [m for m in METODOS if m != "ninguna" and (finales.metodo == m).any()]
    fig, ax = plt.subplots(figsize=(11, 1.6 + 1.3 * len(otros)))
    for i, metodo in enumerate(otros):
        s = finales[finales.metodo == metodo].set_index("semilla")["ganancia_corte_validacion"]
        comunes = s.index.intersection(base.index)
        dif = (s.loc[comunes] - base.loc[comunes]) / 1e6
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(dif))
        ax.scatter(dif, i + jitter, s=60, color=COLOR[metodo], edgecolor=FONDO, linewidth=1.5, zorder=3)
        ax.plot([dif.mean()] * 2, [i - 0.3, i + 0.3], color=TINTA, lw=2.5, zorder=4)
        r = comparar(finales, metodo, "ninguna", "ganancia_corte_validacion")
        p_txt = "" if np.isnan(r["p_wilcoxon"]) else " · p = " + f"{r['p_wilcoxon']:.3f}".replace(".", ",")
        ax.annotate(f"media {millones(r['dif_media'])} · gana en {r['gana_en']}/{r['semillas']}{p_txt}",
                    (1.0, i), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=11, color=TINTA_2)
    ax.axvline(0, color=TINTA_2, lw=1.2, ls="--")
    ax.set_yticks(range(len(otros)), [NOMBRE_FAMILIA[m] for m in otros])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Diferencia de ganancia contra F0 en la misma semilla (millones de $)")
    ax.set_title(f"¿Cuánto le gana cada corrección al baseline nominal? ({MES_TEST})")
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)
    fig.savefig(OUTPUT_DIR / "diferencia_vs_baseline.png", dpi=160)
    plt.close(fig)


def grafico_drift() -> None:
    ruta = OUTPUT_DIR / "drift_psi.csv"
    if not ruta.exists():
        return
    d = pd.read_csv(ruta)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, metodo in enumerate(METODOS):
        s = d[d.metodo == metodo]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(s))
        normal, adm = ~s.administrada.to_numpy(), s.administrada.to_numpy()
        x = np.clip(s.psi.to_numpy(), 1e-4, None)
        ax.scatter(x[normal], (i + jitter)[normal], s=40, color=COLOR[metodo], alpha=0.85, edgecolor=FONDO, lw=1,
                   zorder=3)
        ax.scatter(x[adm], (i + jitter)[adm], s=70, facecolor=FONDO, edgecolor=COLOR[metodo], lw=2, zorder=4)
        med = np.median(x[normal])
        ax.plot([med] * 2, [i - 0.32, i + 0.32], color=TINTA, lw=2.5, zorder=5)
        ax.annotate(f"mediana {med:.3f}".replace(".", ","), (med, i - 0.36), ha="center", va="bottom",
                    fontsize=10.5, color=TINTA_2)
    ax.axvline(0.1, color=TINTA_2, lw=1, ls=":")
    ax.axvline(0.25, color=TINTA_2, lw=1, ls=":")
    ax.text(0.1, -0.72, "moderado 0,1 ", fontsize=10, color=TINTA_2, ha="right")
    ax.text(0.25, -0.72, " 0,25 fuerte", fontsize=10, color=TINTA_2, ha="left")
    ax.set_xscale("log")
    ax.set_yticks(range(len(METODOS)), [NOMBRE_FAMILIA[m] for m in METODOS])
    ax.set_ylim(len(METODOS) - 0.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PSI de cada variable en pesos: 201903-201908 contra 201910 (escala log; < 0,0001 se muestra en 0,0001)")
    ax.set_title("¿Cuánto se movió cada variable en pesos entre entrenamiento y test?", pad=28)
    ax.text(0, 1.02, "Punto lleno = variable de mercado · anillo = límite o comisión que el banco fija por escalones · "
            "raya = mediana de las de mercado", transform=ax.transAxes, fontsize=10.5, color=TINTA_2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drift_psi.png", dpi=160)
    plt.close(fig)


def main() -> None:
    bo, finales, curvas = cargar()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partes = ["# Resumen del experimento (generado por scripts/06_resumen.py)\n",
              f"Ganancia: +{miles(GANANCIA_ACIERTO)} por BAJA+2 estimulado, -{miles(COSTO_ESTIMULO)} por estímulo.\n"]

    if bo:
        filas = []
        for m in METODOS:
            if m in bo:
                b, h = bo[m], bo[m]["hiperparametros"]
                filas.append({"familia": NOMBRE_FAMILIA[m], "AUC 201908": f"{b['auc_validacion']:.4f}",
                              "árboles": b["best_iteration"], "num_leaves": h["num_leaves"],
                              "learning_rate": f"{h['learning_rate']:.4f}", "min_data_in_leaf": h["min_data_in_leaf"],
                              "feature_fraction": f"{h['feature_fraction']:.2f}",
                              "bagging_fraction": f"{h['bagging_fraction']:.2f}",
                              "corte (201908)": miles(b["corte_validacion"]),
                              "ganancia 201908": millones(b["ganancia_validacion"]),
                              "intentos": b["intentos_completos"]})
        partes += ["## Optimización bayesiana (validación 201908)\n", tabla_md(pd.DataFrame(filas)), ""]

    if not finales.empty:
        finales.sort_values(["metodo", "semilla"]).to_csv(OUTPUT_DIR / "resultados_por_semilla.csv", index=False)
        filas = []
        for m in METODOS:
            f = finales[finales.metodo == m]
            if f.empty:
                continue
            fila = {"familia": NOMBRE_FAMILIA[m], "semillas": len(f),
                    "ganancia (corte de validación)": f"{millones(f.ganancia_corte_validacion.mean())} ± "
                                                      f"{millones(f.ganancia_corte_validacion.std())}",
                    "ganancia máxima (optimista)": f"{millones(f.ganancia_maxima.mean())} ± "
                                                   f"{millones(f.ganancia_maxima.std())}",
                    "AUC 201910": f"{f.auc_test.mean():.4f} ± {f.auc_test.std():.4f}"}
            if m != "ninguna" and (finales.metodo == "ninguna").any():
                r = comparar(finales, m, "ninguna", "ganancia_corte_validacion")
                supera = r["dif_media"] > r["desvio_contra"] and r["p_wilcoxon"] < 0.05
                fila.update({"dif. vs F0": millones(r["dif_media"]), "gana en": f"{r['gana_en']}/{r['semillas']}",
                             "p Wilcoxon": f"{r['p_wilcoxon']:.4f}", "¿supera a F0?": "sí" if supera else "no"})
            filas.append(fila)
        partes += [f"## Test {MES_TEST} (reentreno 201903-201908, misma semilla en las 4 familias)\n",
                   tabla_md(pd.DataFrame(filas).fillna("")), ""]

        filas = []
        for m in ["mediana", "ranking"]:
            if (finales.metodo == m).any() and (finales.metodo == "ipc").any():
                r = comparar(finales, m, "ipc", "ganancia_corte_validacion")
                empata = r["dif_media"] >= 0 or not (r["p_wilcoxon"] < 0.05)
                filas.append({"familia": NOMBRE_FAMILIA[m], "dif. vs F1 IPC": millones(r["dif_media"]),
                              "gana en": f"{r['gana_en']}/{r['semillas']}", "p Wilcoxon": f"{r['p_wilcoxon']:.4f}",
                              "¿empata o le gana al IPC?": "sí" if empata else "no"})
        if filas:
            partes += ["## Corrección interna contra IPC\n", tabla_md(pd.DataFrame(filas)), ""]
        grafico_curvas(curvas, bo)
        grafico_diferencias(finales)

    ruta_drift = OUTPUT_DIR / "drift_resumen.csv"
    if ruta_drift.exists():
        d = pd.read_csv(ruta_drift)
        d["familia"] = d.metodo.map(NOMBRE_FAMILIA)
        cols = ["familia", "psi_mediana", "psi_mediana_no_administradas", "psi_mediana_administradas",
                "pct_psi_mayor_0_10", "auc_adversarial_continuas", "auc_adversarial_todas"]
        partes += ["## Drift de las variables en pesos (201903-201908 contra 201910)\n",
                   tabla_md(d[cols].round(3)), ""]
        grafico_drift()

    (OUTPUT_DIR / "resumen.md").write_text("\n".join(partes), encoding="utf-8")
    print("\n".join(partes))


if __name__ == "__main__":
    main()
