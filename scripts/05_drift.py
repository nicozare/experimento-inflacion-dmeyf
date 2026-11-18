"""Mide el drift de las variables en pesos antes y después de cada corrección.

Separa las dos preguntas de la hipótesis: (a) ¿la corrección baja el drift?
se mide acá, sin el modelo de churn; (b) ¿sube la ganancia? se mide en 04.

Referencia: los meses del reentreno final (201903-201908). Comparado con:
201910 (test). Solo las variables en pesos, con su valor del mes (sin lags).
Se excluyen las que tienen alguna celda rota en esos meses (output/celdas_rotas.csv):
valen 0 para todos los clientes un mes entero, en las 4 familias por igual,
y dominarían cualquier medida de drift sin decir nada de la inflación.

Dos medidas:
  PSI (Population Stability Index) por variable: deciles de la referencia
      (sin contar ceros), más un intervalo para el 0 y otro para el nulo.
      Cerca de 0 = no cambió; > 0,1 cambio moderado; > 0,25 cambio fuerte.
  AUC adversarial: un LightGBM intenta distinguir filas de 201910 de filas de
      la referencia usando solo esas variables (validación cruzada de 2
      particiones, muestras balanceadas). 0,5 = indistinguibles (sin drift);
      1 = se distinguen perfecto. Se calcula dos veces:
        - con todas las variables sanas;
        - solo con las continuas (>= 1.000 valores distintos por mes). Las que
          tienen pocos valores (límites de tarjeta y comisiones que el banco
          fija por escalones) quedan, después de deflactar o rankear, con un
          valor exacto distinto en cada mes: el clasificador reconoce el mes
          por esa "huella" (AUC ~1) aunque la distribución no se haya movido.

Las variables "administradas" (límites y comisión de mantenimiento) se marcan
aparte: el banco las ajusta por escalones, no mes a mes con la inflación.

Salida: output/drift_psi.csv (por variable) y output/drift_resumen.csv.

Uso:
    uv run python scripts/05_drift.py
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from comun import COLUMNAS_CLAVE, auc, es_monetaria
from config import MES_TEST, MESES_ENTRENAMIENTO_FINAL, METODOS, OUTPUT_DIR, PARAMS_FIJOS, SEMILLA_BO, ruta_fe

FILAS_POR_LADO = 50_000
EPS = 1e-6
MIN_VALORES_DISTINTOS = 1_000


def es_administrada(columna: str) -> bool:
    """Montos que fija el banco por escalones (no siguen la inflación mes a mes)."""
    return "limite" in columna or columna == "mcomisiones_mantenimiento"


def variables_sanas() -> list[str]:
    esquema = pq.read_schema(ruta_fe("ninguna", MES_TEST))
    monetarias = [c for c in esquema.names
                  if c not in COLUMNAS_CLAVE and not c.startswith(("lag_", "delta_")) and es_monetaria(c)]
    rotas = pd.read_csv(OUTPUT_DIR / "celdas_rotas.csv")
    meses = MESES_ENTRENAMIENTO_FINAL + [MES_TEST]
    excluir = set(rotas.loc[rotas["foto_mes"].isin(meses), "variable"])
    return [c for c in monetarias if c not in excluir]


def leer(metodo: str, meses: list[int], cols: list[str]) -> pd.DataFrame:
    return pd.concat([pq.read_table(ruta_fe(metodo, m), columns=["foto_mes"] + cols).to_pandas() for m in meses],
                     ignore_index=True)


def psi(ref: np.ndarray, test: np.ndarray) -> float:
    def fracciones(x: np.ndarray, bordes: np.ndarray) -> np.ndarray:
        nulo, cero = np.isnan(x), x == 0
        resto = x[~nulo & ~cero]
        conteo = np.histogram(resto, bins=bordes)[0] if len(bordes) > 1 else np.array([len(resto)])
        return np.concatenate([[nulo.sum(), cero.sum()], conteo]) / len(x)

    no_cero = ref[~np.isnan(ref) & (ref != 0)]
    if len(no_cero):
        bordes = np.unique(np.quantile(no_cero, np.linspace(0, 1, 11)))
        bordes[0], bordes[-1] = -np.inf, np.inf
    else:
        bordes = np.array([-np.inf, np.inf])
    r = np.clip(fracciones(ref, bordes), EPS, None)
    t = np.clip(fracciones(test, bordes), EPS, None)
    return float(np.sum((t - r) * np.log(t / r)))


def auc_adversarial(ref: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    rng = np.random.default_rng(SEMILLA_BO)
    # referencia: la misma cantidad de filas de cada mes
    por_mes = FILAS_POR_LADO // ref["foto_mes"].nunique()
    idx_ref = np.concatenate([rng.choice(np.flatnonzero(ref["foto_mes"].to_numpy() == m), por_mes, replace=False)
                              for m in sorted(ref["foto_mes"].unique())])
    idx_test = rng.choice(len(test), FILAS_POR_LADO, replace=False)
    X = np.vstack([ref[cols].to_numpy(np.float32)[idx_ref], test[cols].to_numpy(np.float32)[idx_test]])
    y = np.concatenate([np.zeros(len(idx_ref)), np.ones(len(idx_test))])
    pliegue = rng.integers(0, 2, len(y))
    pred = np.empty(len(y))
    params = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "seed": SEMILLA_BO,
              "deterministic": True, "force_row_wise": True, "verbosity": -1,
              "num_threads": PARAMS_FIJOS["num_threads"]}
    for k in (0, 1):
        entrena = pliegue != k
        modelo = lgb.train(params, lgb.Dataset(X[entrena], label=y[entrena]), num_boost_round=100)
        pred[~entrena] = modelo.predict(X[~entrena])
    return auc(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", default=METODOS, choices=METODOS)
    args = parser.parse_args()
    cols = variables_sanas()
    print(f"Variables en pesos sin celdas rotas en {MESES_ENTRENAMIENTO_FINAL[0]}-{MES_TEST}: {len(cols)}", flush=True)
    # valores distintos por mes, medidos en nominal (la corrección no los cambia)
    nominal = leer("ninguna", MESES_ENTRENAMIENTO_FINAL, cols)
    distintos = {c: np.mean([nominal.loc[nominal["foto_mes"] == m, c].replace(0, np.nan).nunique()
                             for m in MESES_ENTRENAMIENTO_FINAL]) for c in cols}
    continuas = [c for c in cols if distintos[c] >= MIN_VALORES_DISTINTOS]
    print(f"  continuas (>= {MIN_VALORES_DISTINTOS:,} valores distintos por mes): {len(continuas)}; "
          f"administradas: {[c for c in cols if es_administrada(c)]}", flush=True)

    filas_psi, filas_resumen = [], []
    for metodo in args.metodos:
        ref = leer(metodo, MESES_ENTRENAMIENTO_FINAL, cols)
        test = leer(metodo, [MES_TEST], cols)
        valores = {c: psi(ref[c].to_numpy(np.float64), test[c].to_numpy(np.float64)) for c in cols}
        filas_psi += [{"metodo": metodo, "variable": c, "psi": v, "administrada": es_administrada(c),
                       "continua": c in continuas, "valores_distintos_por_mes": round(distintos[c])}
                      for c, v in valores.items()]
        serie = pd.Series(valores)
        resto = serie[[c for c in cols if not es_administrada(c)]]
        resumen = {
            "metodo": metodo,
            "variables": len(cols),
            "psi_mediana": serie.median(),
            "psi_p90": serie.quantile(0.9),
            "pct_psi_mayor_0_10": (serie > 0.10).mean() * 100,
            "psi_mediana_administradas": serie[[c for c in cols if es_administrada(c)]].median(),
            "psi_mediana_no_administradas": resto.median(),
            "auc_adversarial_todas": auc_adversarial(ref, test, cols),
            "auc_adversarial_continuas": auc_adversarial(ref, test, continuas),
        }
        filas_resumen.append(resumen)
        print(f"  {metodo:8s}: PSI mediana {resumen['psi_mediana']:.3f} (administradas "
              f"{resumen['psi_mediana_administradas']:.3f}) | {resumen['pct_psi_mayor_0_10']:.0f}% con PSI > 0,1 | "
              f"AUC adversarial todas {resumen['auc_adversarial_todas']:.3f}, "
              f"continuas {resumen['auc_adversarial_continuas']:.3f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas_psi).to_csv(OUTPUT_DIR / "drift_psi.csv", index=False)
    pd.DataFrame(filas_resumen).to_csv(OUTPUT_DIR / "drift_resumen.csv", index=False)
    print(f"-> {OUTPUT_DIR / 'drift_resumen.csv'}", flush=True)


if __name__ == "__main__":
    main()
