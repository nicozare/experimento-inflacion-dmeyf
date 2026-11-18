"""S6: cuánto pesa el azar de los clientes del mes (bootstrap).

Las 20 semillas miden el azar del modelo con los clientes de 201910 fijos.
Pero qué clientes caen en el mes también es azar: con otros clientes
parecidos, la ganancia sería otra. Acá se remuestrean con reposición los
clientes de 201910 (el mismo remuestreo para todas las familias y semillas),
se recalcula la ganancia de cada semilla en su corte y se promedia entre
semillas. Para cada familia: intervalo del 95% de su diferencia contra F0,
con el corte de la bayesiana y con el corte estable (S1, de 08).

Regla (README, escrita antes de correr): una diferencia es robusta al azar
de los clientes si su intervalo del 95% no incluye el 0.

Detalle de la cuenta: en un remuestreo, cada cliente aparece w veces
(w ~ Poisson(1)); estimular a los k de mayor probabilidad es recorrer el
orden de la semilla sumando copias hasta llegar a k. Con w = 1 para todos
se recupera exactamente la ganancia de 04 (control que se imprime).

Necesita las predicciones locales de 04 (resultados/prediccion__*.parquet) y
resultados/corte_estable_y_ensamble.csv (08).

Salida:
  output/bootstrap_clientes.md
  output/bootstrap_clientes.png
  resultados/bootstrap_clientes.csv    diferencia de cada comparación en cada remuestreo

Uso:
    uv run python scripts/10_bootstrap_clientes.py                  # 1.000 remuestreos
    uv run python scripts/10_bootstrap_clientes.py --remuestreos 200
"""
from __future__ import annotations

import argparse
import importlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import leer_bo
from config import COSTO_ESTIMULO, GANANCIA_ACIERTO, MES_TEST, METODOS, NOMBRE_FAMILIA, OUTPUT_DIR, RESULTADOS_DIR, SEMILLAS

r6 = importlib.import_module("06_resumen")  # formato, colores y comparación pareada del resumen
r8 = importlib.import_module("08_corte_estable_y_ensamble")  # lectura de predicciones y clase
millones, miles_txt, tabla_md = r6.millones, r6.miles, r6.tabla_md
COLOR, FONDO, TINTA, TINTA_2 = r6.COLOR, r6.FONDO, r6.TINTA, r6.TINTA_2

SEMILLA_BOOTSTRAP = 100081
# Solo importan los primeros del orden: con cortes de hasta 5.000 clientes,
# los 8.000 de mayor probabilidad alcanzan siempre (la suma de 8.000 copias
# Poisson(1) no baja de 5.000 en la práctica; se verifica en cada remuestreo).
TOPE = 8_000
REGLAS = {"bayesiana": "corte de la bayesiana", "estable": "corte estable (S1)"}


def ganancias(orden: np.ndarray, valor: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """Ganancia de cada semilla (filas de `orden`) estimulando a los k primeros,
    con cada cliente repetido w veces."""
    W = w[orden]
    hasta = np.cumsum(W, axis=1)
    if (hasta[:, -1] < k).any():
        raise SystemExit("TOPE chico para este corte: subirlo")
    copias = np.clip(k - (hasta - W), 0, W)
    return (copias * valor[orden]).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remuestreos", type=int, default=1000)
    B = parser.parse_args().remuestreos
    t0 = time.time()

    familias = [m for m in METODOS
                if all((RESULTADOS_DIR / f"prediccion__{m}__{s}.parquet").exists() for s in SEMILLAS)]
    if "ninguna" not in familias:
        raise SystemExit("Faltan las predicciones de F0: correr antes scripts/04_entrenar_final.py")
    estable = pd.read_csv(RESULTADOS_DIR / "corte_estable_y_ensamble.csv")
    estable = estable[estable.grupo == "S1"].groupby("metodo").corte.first()
    familias = [m for m in familias if m in estable.index]

    clientes, orden, cortes = None, {}, {}
    for m in familias:
        c, probs = r8.predicciones("prediccion", m, SEMILLAS)
        if clientes is None:
            clientes = c
        elif not np.array_equal(clientes, c):
            raise SystemExit(f"{m}: los clientes no están en el mismo orden que en F0")
        orden[m] = np.stack([np.argsort(-probs[s], kind="stable")[:TOPE] for s in SEMILLAS])
        cortes[m] = {"bayesiana": int(leer_bo(m)["corte_validacion"]), "estable": int(estable[m])}
    clase = r8.clase_de(MES_TEST, clientes)
    valor = np.where(clase == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO).astype(np.int64)
    n = len(clientes)

    # comparaciones: cada familia contra F0, y las de la hipótesis secundaria y de S8
    comparaciones = [(m, "ninguna") for m in familias if m != "ninguna"]
    comparaciones += [(a, b) for a, b in [("mediana", "ipc"), ("ranking", "ipc"), ("proporciones", "diferencias")]
                      if a in familias and b in familias]

    def medias(w: np.ndarray) -> dict:
        return {(m, r): ganancias(orden[m], valor, w, cortes[m][r]).mean() for m in familias for r in REGLAS}

    # control: sin remuestrear (w = 1) tiene que dar lo mismo que 04 y 08
    observada = medias(np.ones(n, dtype=np.int64))
    finales = pd.read_csv(OUTPUT_DIR / "resultados_por_semilla.csv")
    s1 = pd.read_csv(RESULTADOS_DIR / "corte_estable_y_ensamble.csv")
    for m in familias:
        a = finales[finales.metodo == m].ganancia_corte_validacion.mean()
        b = s1[(s1.grupo == "S1") & (s1.metodo == m)].ganancia.mean()
        if not (np.isclose(observada[(m, "bayesiana")], a) and np.isclose(observada[(m, "estable")], b)):
            raise SystemExit(f"{m}: sin remuestrear no da la ganancia de 04/08")
    print(f"control: sin remuestrear se recupera la ganancia de 04 y 08 en las {len(familias)} familias", flush=True)

    rng = np.random.default_rng(SEMILLA_BOOTSTRAP)
    filas = []
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        g = medias(w)
        filas += [{"remuestreo": b, "familia": a, "contra": c, "regla": r, "diferencia": g[(a, r)] - g[(c, r)]}
                  for a, c in comparaciones for r in REGLAS]
        if (b + 1) % 200 == 0:
            print(f"  {b + 1} remuestreos ({time.time() - t0:.0f} s)", flush=True)
    boot = pd.DataFrame(filas)
    boot.to_csv(RESULTADOS_DIR / "bootstrap_clientes.csv", index=False)

    # error estándar de la diferencia media solo por semillas (clientes fijos), para comparar
    curvas = {}
    for m in familias:
        curvas[m] = {r: ganancias(orden[m], valor, np.ones(n, dtype=np.int64), cortes[m][r]) for r in REGLAS}
    resumen = []
    for a, c in comparaciones:
        for r, nombre_regla in REGLAS.items():
            d = boot[(boot.familia == a) & (boot.contra == c) & (boot.regla == r)].diferencia
            lo, hi = np.percentile(d, [2.5, 97.5])
            por_semilla = curvas[a][r] - curvas[c][r]
            resumen.append({
                "comparación": f"{NOMBRE_FAMILIA[a]} vs {NOMBRE_FAMILIA[c]}", "familia": a, "contra": c,
                "regla": nombre_regla, "regla_clave": r,
                "diferencia": observada[(a, r)] - observada[(c, r)], "ic_inferior": lo, "ic_superior": hi,
                "pct_positiva": 100 * (d > 0).mean(), "ee_clientes": d.std(),
                "ee_semillas": por_semilla.std(ddof=1) / np.sqrt(len(por_semilla)),
            })
    res = pd.DataFrame(resumen)

    tabla = pd.DataFrame({
        "comparación": res["comparación"], "corte": res.regla,
        "diferencia observada": res.diferencia.map(millones),
        "intervalo del 95%": [f"{millones(lo)} a {millones(hi)}" for lo, hi in zip(res.ic_inferior, res.ic_superior)],
        "remuestreos con diferencia > 0": res.pct_positiva.map(lambda v: f"{v:.0f}%"),
        "error estándar: clientes": res.ee_clientes.map(millones),
        "error estándar: semillas": res.ee_semillas.map(millones),
        "¿robusta al azar de los clientes?": [("sí" if lo > 0 or hi < 0 else "no")
                                               for lo, hi in zip(res.ic_inferior, res.ic_superior)],
    })
    partes = ["# Bootstrap de clientes (generado por scripts/10_bootstrap_clientes.py)\n",
              f"Seguimiento S6 (regla en el README). {miles_txt(B)} remuestreos con reposición de los clientes de "
              f"{MES_TEST}, los mismos para todas las familias y semillas; en cada uno, ganancia media de las "
              f"{len(SEMILLAS)} semillas en el corte de cada familia. El intervalo no incluye el azar de las "
              "semillas: es el que mide \"error estándar: semillas\" (la diferencia media con los clientes fijos, "
              "lo que ya medía el Wilcoxon).\n",
              tabla_md(tabla), ""]
    (OUTPUT_DIR / "bootstrap_clientes.md").write_text("\n".join(partes), encoding="utf-8")
    grafico(res, B)
    print("\n".join(partes))
    print(f"-> output/bootstrap_clientes.md, .png y resultados/bootstrap_clientes.csv ({time.time() - t0:.0f} s)")


def grafico(res: pd.DataFrame, B: int) -> None:
    contra_f0 = res[res.contra == "ninguna"]
    familias = list(dict.fromkeys(contra_f0.familia))
    fig, ax = plt.subplots(figsize=(11, 1.7 + 1.0 * len(familias)))
    for i, m in enumerate(familias):
        for j, (r, lleno) in enumerate([("bayesiana", True), ("estable", False)]):
            f = contra_f0[(contra_f0.familia == m) & (contra_f0.regla_clave == r)].iloc[0]
            y = i + (-0.16 if j == 0 else 0.16)
            ax.plot([f.ic_inferior / 1e6, f.ic_superior / 1e6], [y, y], color=COLOR[m], lw=2.2, solid_capstyle="round")
            ax.plot([f.diferencia / 1e6], [y], "o", ms=9, mew=2, mec=COLOR[m], mfc=COLOR[m] if lleno else FONDO,
                    zorder=4)
    ax.axvline(0, color=TINTA_2, lw=1.2, ls="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    ax.set_yticks(range(len(familias)), [NOMBRE_FAMILIA[m] for m in familias])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.plot([], [], "o", ms=9, mew=2, mec=TINTA_2, mfc=TINTA_2, label="corte de la bayesiana")
    ax.plot([], [], "o", ms=9, mew=2, mec=TINTA_2, mfc=FONDO, label="corte estable (S1)")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.30 if len(familias) <= 4 else -0.26))
    ax.set_xlabel(f"Ganancia de la familia menos la de F0 en {MES_TEST} (millones de $)")
    ax.set_title(f"Diferencia contra F0 con el azar de los clientes de {MES_TEST}", pad=50)
    ax.text(0, 1.025, f"Punto = diferencia observada (media de {len(SEMILLAS)} semillas). Barra = intervalo del 95% "
            f"en {miles_txt(B)} remuestreos de clientes.\nSi la barra cruza el 0, la diferencia no es robusta al azar "
            "de qué clientes caen en el mes.", transform=ax.transAxes, fontsize=11, color=TINTA_2,
            va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bootstrap_clientes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
