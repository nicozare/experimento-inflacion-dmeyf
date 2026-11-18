"""Modelo final por familia: reentrena con 201903-201908 y mide en 201910.

Para cada familia toma los hiperparámetros de su bayesiana
(resultados/bo__<metodo>.json), reentrena con 201903-201908 (respuesta de
Ale: para el test se reentrena incluyendo 201907 y 201908) con la cantidad
de árboles que encontró el early stopping, una vez por semilla, y predice
201910. Las 20 semillas son las mismas en las 4 familias: la comparación es
pareada por semilla.

Para cada semilla se guarda:
  - la curva de ganancia en 201910 (de config.CORTE_MINIMO a CORTE_MAXIMO),
  - la ganancia en el corte que eligió la bayesiana en 201908 (métrica
    principal: el corte no se elige mirando el test),
  - la ganancia máxima de la curva (optimista: elige el corte mirando el test),
  - el AUC en 201910 (BAJA+1 y BAJA+2 como positivo, igual que en la validación).

Salida:
  resultados/final__<metodo>__<semilla>.json   métricas de la semilla (se versiona)
  resultados/curva__<metodo>__<semilla>.csv    curva de ganancia (se versiona)
  resultados/prediccion__<metodo>__<semilla>.parquet  probabilidades por cliente (local)

Retoma solo: si el json de una semilla ya existe, la saltea (--forzar para rehacer).

Uso:
    uv run python scripts/04_entrenar_final.py
    uv run python scripts/04_entrenar_final.py --metodos ranking --semillas 100081 100103
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from comun import (
    auc,
    cargar_meses,
    columnas_features,
    cortes_barrido,
    etiqueta_binaria,
    ganancia_por_corte,
    leer_bo,
)
from config import MES_TEST, MESES_ENTRENAMIENTO_FINAL, METODOS, PARAMS_FIJOS, RESULTADOS_DIR, SEMILLAS


def entrenar_familia(metodo: str, semillas: list[int], forzar: bool) -> None:
    pendientes = [s for s in semillas if forzar or not (RESULTADOS_DIR / f"final__{metodo}__{s}.json").exists()]
    if not pendientes:
        print(f"\n=== {metodo}: las {len(semillas)} semillas ya están hechas ===", flush=True)
        return
    bo = leer_bo(metodo)
    feats = columnas_features(metodo, MES_TEST)
    X_train, k_train = cargar_meses(metodo, MESES_ENTRENAMIENTO_FINAL, feats)
    X_test, k_test = cargar_meses(metodo, [MES_TEST], feats)
    clase_test = k_test["clase_ternaria"].to_numpy()
    y_test = etiqueta_binaria(k_test["clase_ternaria"])
    cortes = cortes_barrido()
    corte_val = int(bo["corte_validacion"])
    i_corte = int(np.searchsorted(cortes, corte_val))
    print(f"\n=== {metodo}: entrena {MESES_ENTRENAMIENTO_FINAL} ({len(X_train):,} filas), "
          f"{bo['best_iteration']} árboles, corte de validación {corte_val:,}; "
          f"{len(pendientes)} semillas pendientes ===", flush=True)

    params_dataset = {k: PARAMS_FIJOS[k] for k in ("feature_pre_filter", "verbosity", "num_threads")}
    dtrain = lgb.Dataset(X_train, label=etiqueta_binaria(k_train["clase_ternaria"]),
                         feature_name=feats, params=params_dataset, free_raw_data=False).construct()

    for semilla in pendientes:
        inicio = time.time()
        params = {**PARAMS_FIJOS, **bo["hiperparametros"], "seed": semilla}
        modelo = lgb.train(params, dtrain, num_boost_round=int(bo["best_iteration"]))
        prob = modelo.predict(X_test)
        g = ganancia_por_corte(prob, clase_test, cortes)
        metricas = {
            "metodo": metodo,
            "semilla": semilla,
            "corte_validacion": corte_val,
            "ganancia_corte_validacion": int(g[i_corte]),
            "ganancia_maxima": int(g.max()),
            "corte_ganancia_maxima": int(cortes[np.argmax(g)]),
            "auc_test": auc(y_test, prob),
            "arboles": int(bo["best_iteration"]),
            "segundos": round(time.time() - inicio, 1),
            "lightgbm": lgb.__version__,
            "maquina": f"{platform.system()} {platform.machine()}",
        }
        pd.DataFrame({"corte": cortes, "ganancia": g}).to_csv(
            RESULTADOS_DIR / f"curva__{metodo}__{semilla}.csv", index=False)
        pd.DataFrame({"numero_de_cliente": k_test["numero_de_cliente"].to_numpy(), "prob": prob}).to_parquet(
            RESULTADOS_DIR / f"prediccion__{metodo}__{semilla}.parquet", index=False)
        (RESULTADOS_DIR / f"final__{metodo}__{semilla}.json").write_text(
            json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  semilla {semilla}: ganancia ${metricas['ganancia_corte_validacion'] / 1e6:,.1f}M "
              f"en {corte_val:,} | máxima ${metricas['ganancia_maxima'] / 1e6:,.1f}M "
              f"en {metricas['corte_ganancia_maxima']:,} | AUC {metricas['auc_test']:.4f} | "
              f"{metricas['segundos']:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", default=METODOS, choices=METODOS)
    parser.add_argument("--semillas", nargs="+", type=int, default=SEMILLAS)
    parser.add_argument("--forzar", action="store_true", help="rehace las semillas que ya tienen resultado")
    args = parser.parse_args()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    for metodo in args.metodos:
        entrenar_familia(metodo, args.semillas, args.forzar)


if __name__ == "__main__":
    main()
