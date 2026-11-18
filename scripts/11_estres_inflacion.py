"""S7: cuánta inflación hace falta para que corregir convenga (prueba de estrés).

Con 2019, entre el entrenamiento y octubre hay ~18% de inflación, y corregir
no conviene. Acá se infla artificialmente octubre: todos los montos en pesos
(nivel, lags y deltas de las 73 variables) se multiplican por k, como si el
test estuviera más lejos del entrenamiento, y se mide cuánto pierden F0 y F4
con su corte de la bayesiana. Con una inflación pareja, F2 y F3 no cambian
por construcción (la mediana y el ranking del mes la absorben), y F1 tampoco
si el IPC la mide: el punto en que F0 cae por debajo de cada una dice desde
cuánta inflación conviene corregir. Sin regla de decisión.

En F4 se inflan solo sus columnas nominales: las proporciones no cambian
(numerador y denominador suben igual).

Supuesto fuerte: la inflación es pareja para todas las variables. En la
realidad no lo es (README, "Por qué tendría que funcionar") y los límites
van por escalones.

Los modelos son los finales de 04 (mismos hiperparámetros, árboles y
semillas, Dataset armado igual): con k = 1 tienen que dar exactamente la
ganancia de 04 (control que se imprime). Se guardan en data/modelos/ (no se
versiona) para que 12 no los vuelva a entrenar.

Salida:
  output/estres_inflacion.md
  output/estres_inflacion.png
  resultados/estres_inflacion.csv   ganancia por familia, semilla y factor

Uso:
    uv run python scripts/11_estres_inflacion.py                    # 5 semillas
    uv run python scripts/11_estres_inflacion.py --semillas 100081
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import cargar_meses, columnas_features, es_monetaria, etiqueta_binaria, ganancia_por_corte, leer_bo
from config import (DATA_DIR, IPC, MES_TEST, MESES_ENTRENAMIENTO_FINAL, NOMBRE_FAMILIA, OUTPUT_DIR, PARAMS_FIJOS,
                    RESULTADOS_DIR, SEMILLAS)

r6 = importlib.import_module("06_resumen")  # formato y colores del resumen
separar = importlib.import_module("09_donde_corta").separar  # nombre base y tipo (nivel, lag, delta) de una columna
millones, miles, decimal, tabla_md = r6.millones, r6.miles, r6.decimal, r6.tabla_md
COLOR, FONDO, TINTA, TINTA_2 = r6.COLOR, r6.FONDO, r6.TINTA, r6.TINTA_2

FAMILIAS = ["ninguna", "proporciones"]
FACTORES = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
MODELOS_DIR = DATA_DIR / "modelos"
# inflación que ya hay entre el mes promedio del entrenamiento y el test
INFLACION_BASE = IPC[MES_TEST] / np.mean([IPC[m] for m in MESES_ENTRENAMIENTO_FINAL])


def es_pesos(feature: str) -> bool:
    """Nivel, lag o delta de una variable en pesos (o de una diferencia de F5,
    que también está en pesos). Las proporciones de F4 no."""
    base = separar(feature)[0]
    return es_monetaria(base) or base.startswith("dif_")


def ruta_modelo(metodo: str, semilla: int):
    return MODELOS_DIR / f"{metodo}__{semilla}.txt"


def modelos_finales(metodo: str, semillas: list[int], X_train, k_train, feats) -> dict[int, lgb.Booster]:
    """Los modelos finales de 04, entrenados igual (o leídos de data/modelos/)."""
    bo = leer_bo(metodo)
    modelos, dtrain = {}, None
    for s in semillas:
        ruta = ruta_modelo(metodo, s)
        if ruta.exists():
            modelos[s] = lgb.Booster(model_file=str(ruta))
            continue
        if dtrain is None:
            # igual que 04: el dataset se arma ANTES de entrenar y sin la semilla
            dtrain = lgb.Dataset(X_train, label=etiqueta_binaria(k_train["clase_ternaria"]), feature_name=feats,
                                 free_raw_data=False,
                                 params={c: PARAMS_FIJOS[c] for c in ("feature_pre_filter", "verbosity",
                                                                      "num_threads")}).construct()
        inicio = time.time()
        modelos[s] = lgb.train({**PARAMS_FIJOS, **bo["hiperparametros"], "seed": s}, dtrain,
                               num_boost_round=int(bo["best_iteration"]))
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        modelos[s].save_model(str(ruta))
        print(f"  {metodo} semilla {s}: entrenado ({time.time() - inicio:.0f} s)", flush=True)
    return modelos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--semillas", nargs="+", type=int, default=SEMILLAS[:5])
    semillas = parser.parse_args().semillas
    t0 = time.time()

    filas = []
    for metodo in FAMILIAS:
        bo = leer_bo(metodo)
        corte = int(bo["corte_validacion"])
        feats = columnas_features(metodo, MES_TEST)
        if all(ruta_modelo(metodo, s).exists() for s in semillas):  # ya entrenados: no hace falta el entrenamiento
            X_train, k_train = None, None
        else:
            X_train, k_train = cargar_meses(metodo, MESES_ENTRENAMIENTO_FINAL, feats)
        X_test, k_test = cargar_meses(metodo, [MES_TEST], feats)
        clase = k_test["clase_ternaria"].to_numpy()
        pesos = [j for j, f in enumerate(feats) if es_pesos(f)]
        print(f"\n=== {metodo}: {len(pesos)} de {len(feats)} columnas en pesos se inflan; corte {miles(corte)} ===",
              flush=True)
        modelos = modelos_finales(metodo, semillas, X_train, k_train, feats)
        del X_train
        for s, modelo in modelos.items():
            base = None
            for k in FACTORES:
                X = X_test.copy()
                X[:, pesos] *= np.float32(k)
                prob = modelo.predict(X)
                g = int(ganancia_por_corte(prob, clase, np.array([corte]))[0])
                elegidos = set(np.argsort(-prob, kind="stable")[:corte])
                if base is None:
                    base = elegidos
                    final = json.loads((RESULTADOS_DIR / f"final__{metodo}__{s}.json").read_text(encoding="utf-8"))
                    if g != final["ganancia_corte_validacion"]:
                        raise SystemExit(f"{metodo} semilla {s}: con k = 1 no da la ganancia de 04")
                filas.append({"metodo": metodo, "semilla": s, "factor": k, "corte": corte, "ganancia": g,
                              "mismos_clientes": len(base & elegidos) / corte})
            print(f"  {metodo} semilla {s}: " + " | ".join(
                f"x{decimal(f['factor'], 2)} {millones(f['ganancia'])}" for f in filas[-len(FACTORES):]), flush=True)
        del X_test
    print(f"\ncontrol: con k = 1 se recupera la ganancia de 04 en las {len(semillas)} semillas de F0 y F4", flush=True)
    estres = pd.DataFrame(filas)
    estres.to_csv(RESULTADOS_DIR / "estres_inflacion.csv", index=False)

    # Para comparar con las 20 semillas de cada familia: la pérdida de cada k
    # (medida en estas semillas) se suma a la ganancia media de las 20.
    finales = pd.read_csv(OUTPUT_DIR / "resultados_por_semilla.csv")
    media20 = finales.groupby("metodo").ganancia_corte_validacion.mean()
    perdida = (estres.groupby(["metodo", "factor"]).ganancia.mean()
               - estres[estres.factor == 1.0].groupby("metodo").ganancia.mean())
    curva = {m: media20[m] + perdida[m] for m in FAMILIAS}
    referencias = [m for m in ["ipc", "mediana", "ranking"] if m in media20.index]

    def cruce(m: str, nivel: float) -> float | None:
        """Inflación total (entrenamiento -> test) desde la que la familia m queda por debajo de `nivel`."""
        c = curva[m]
        total = INFLACION_BASE * c.index.to_numpy() - 1
        abajo = np.nonzero(c.to_numpy() < nivel)[0]
        if not len(abajo):
            return None
        i = abajo[0]
        if i == 0:
            return total[0]
        y0, y1 = c.iloc[i - 1], c.iloc[i]
        return total[i - 1] + (total[i] - total[i - 1]) * (y0 - nivel) / (y0 - y1)

    tabla = []
    for m in FAMILIAS:
        fila = {"familia": NOMBRE_FAMILIA[m]}
        for k in FACTORES:
            mismos = estres[(estres.metodo == m) & (estres.factor == k)].mismos_clientes.mean()
            fila[f"x{decimal(k, 2)} (+{(INFLACION_BASE * k - 1) * 100:.0f}%)"] = (
                f"{millones(curva[m][k])} ({100 * mismos:.0f}%)")
        tabla.append(fila)
    cruces = []
    for m in FAMILIAS:
        for ref in referencias:
            x = cruce(m, media20[ref])
            cruces.append({"familia": NOMBRE_FAMILIA[m], "queda por debajo de": NOMBRE_FAMILIA[ref],
                           "su ganancia (no cambia)": millones(media20[ref]),
                           "desde una inflación total de": "no en este rango" if x is None else f"+{100 * x:.0f}%"})
    partes = ["# Prueba de estrés de inflación (generado por scripts/11_estres_inflacion.py)\n",
              f"Seguimiento S7 (README). Montos en pesos de {MES_TEST} multiplicados por k (nivel, lags y deltas). "
              f"Entre paréntesis en cada columna, la inflación total entre el mes promedio del entrenamiento y el test "
              f"(hoy +{(INFLACION_BASE - 1) * 100:.0f}%). Ganancia en el corte de la bayesiana: la pérdida medida en "
              f"{len(semillas)} semillas, sumada a la media de las 20; entre paréntesis en cada celda, qué parte de "
              "los clientes elegidos son los mismos que sin inflar.\n",
              tabla_md(pd.DataFrame(tabla)), "",
              "## Desde cuánta inflación conviene corregir\n",
              "Con una inflación pareja, F1 (si el IPC la mide), F2 y F3 no cambian: su ganancia es la de las 20 "
              "semillas. \"Desde\" es el primer cruce, interpolando entre los factores probados.\n",
              tabla_md(pd.DataFrame(cruces)), ""]
    (OUTPUT_DIR / "estres_inflacion.md").write_text("\n".join(partes), encoding="utf-8")
    grafico(curva, media20, referencias)
    print("\n".join(partes))
    print(f"-> output/estres_inflacion.md, .png y resultados/estres_inflacion.csv ({time.time() - t0:.0f} s)")


def grafico(curva: dict, media20: pd.Series, referencias: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for ref in referencias:
        ax.axhline(media20[ref] / 1e6, color=COLOR[ref], lw=1.8, ls="--", zorder=2)
        ax.annotate(f"{NOMBRE_FAMILIA[ref]}: no cambia", (1.0, media20[ref] / 1e6), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=11, color=TINTA_2)
    for m in FAMILIAS:
        c = curva[m]
        x = (INFLACION_BASE * c.index.to_numpy() - 1) * 100
        ax.plot(x, c.to_numpy() / 1e6, "-o", color=COLOR[m], lw=2.2, ms=8, mec=FONDO, mew=1.5, zorder=4,
                label=NOMBRE_FAMILIA[m])
    ax.axvline((INFLACION_BASE - 1) * 100, color=TINTA_2, lw=1, ls=":", zorder=1)
    ax.annotate("2019 real", ((INFLACION_BASE - 1) * 100, 0.0), xycoords=("data", "axes fraction"),
                xytext=(5, 6), textcoords="offset points", va="bottom", fontsize=10.5, color=TINTA_2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    ax.set_xlabel("Inflación total entre el mes promedio del entrenamiento y el test")
    ax.set_ylabel(f"Ganancia en {MES_TEST} (millones de $)")
    ax.set_title("¿Desde cuánta inflación conviene corregir?", pad=50)
    ax.text(0, 1.025, f"Montos en pesos de {MES_TEST} inflados artificialmente, parejo para todas las variables.\n"
            "F1, F2 y F3 no cambian por construcción (F1, si el IPC mide esa inflación).", transform=ax.transAxes,
            fontsize=11, color=TINTA_2, va="bottom")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.27))
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUTPUT_DIR / "estres_inflacion.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
