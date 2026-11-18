"""Las 20 semillas de cada familia en validación (experimento de seguimiento).

Para el corte estable y el ensamble (08) hace falta la predicción de 201908 de
cada semilla, no solo la del mejor intento de la bayesiana. Este script
entrena cada familia con 201903-201906 (como en la bayesiana), con sus
hiperparámetros y su cantidad de árboles, una vez por semilla, y predice
201908. 201910 no se toca acá.

Control: la primera semilla es la de la bayesiana, así que tiene que
reproducir el AUC y el corte de su mejor intento (se avisa si no).

Salida:
  resultados/validacion__<metodo>__<semilla>.json          AUC y ganancia en 201908 (se versiona)
  resultados/curva_validacion__<metodo>__<semilla>.csv     curva de ganancia en 201908 (se versiona)
  resultados/prediccion_validacion__<metodo>__<semilla>.parquet  probabilidades (local)

Retoma solo: si el json de una semilla ya existe, la saltea (--forzar para rehacer).

Uso:
    uv run python scripts/07_validacion_semillas.py
    uv run python scripts/07_validacion_semillas.py --metodos proporciones --semillas 100081 100103
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from comun import auc, cargar_meses, columnas_features, cortes_barrido, etiqueta_binaria, ganancia_por_corte, leer_bo
from config import MES_VALIDACION, MESES_ENTRENAMIENTO, METODOS, PARAMS_FIJOS, RESULTADOS_DIR, SEMILLAS


def validar_familia(metodo: str, semillas: list[int], forzar: bool) -> None:
    pendientes = [s for s in semillas
                  if forzar or not (RESULTADOS_DIR / f"validacion__{metodo}__{s}.json").exists()]
    if not pendientes:
        print(f"\n=== {metodo}: las {len(semillas)} semillas ya están hechas ===", flush=True)
        return
    bo = leer_bo(metodo)
    feats = columnas_features(metodo, MES_VALIDACION)
    X_train, k_train = cargar_meses(metodo, MESES_ENTRENAMIENTO, feats)
    X_valid, k_valid = cargar_meses(metodo, [MES_VALIDACION], feats)
    clase_valid = k_valid["clase_ternaria"].to_numpy()
    y_valid = etiqueta_binaria(k_valid["clase_ternaria"])
    cortes = cortes_barrido()
    print(f"\n=== {metodo}: entrena {MESES_ENTRENAMIENTO} ({len(X_train):,} filas), predice {MES_VALIDACION}, "
          f"{bo['best_iteration']} árboles; {len(pendientes)} semillas pendientes ===", flush=True)

    # igual que 03 y 04: el dataset se arma una vez, antes de entrenar y sin la semilla
    params_dataset = {k: PARAMS_FIJOS[k] for k in ("feature_pre_filter", "verbosity", "num_threads")}
    dtrain = lgb.Dataset(X_train, label=etiqueta_binaria(k_train["clase_ternaria"]),
                         feature_name=feats, params=params_dataset, free_raw_data=False).construct()

    for semilla in pendientes:
        inicio = time.time()
        modelo = lgb.train({**PARAMS_FIJOS, **bo["hiperparametros"], "seed": semilla}, dtrain,
                           num_boost_round=int(bo["best_iteration"]))
        prob = modelo.predict(X_valid)
        g = ganancia_por_corte(prob, clase_valid, cortes)
        metricas = {
            "metodo": metodo,
            "semilla": semilla,
            "auc_validacion": auc(y_valid, prob),
            "ganancia_maxima_validacion": int(g.max()),
            "corte_ganancia_maxima_validacion": int(cortes[np.argmax(g)]),
            "arboles": int(bo["best_iteration"]),
            "segundos": round(time.time() - inicio, 1),
            "lightgbm": lgb.__version__,
            "maquina": f"{platform.system()} {platform.machine()}",
        }
        pd.DataFrame({"corte": cortes, "ganancia": g}).to_csv(
            RESULTADOS_DIR / f"curva_validacion__{metodo}__{semilla}.csv", index=False)
        pd.DataFrame({"numero_de_cliente": k_valid["numero_de_cliente"].to_numpy(), "prob": prob}).to_parquet(
            RESULTADOS_DIR / f"prediccion_validacion__{metodo}__{semilla}.parquet", index=False)
        (RESULTADOS_DIR / f"validacion__{metodo}__{semilla}.json").write_text(
            json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        control = ""
        if semilla == bo["semilla_bo"]:
            igual = (abs(metricas["auc_validacion"] - bo["auc_validacion"]) < 1e-5
                     and metricas["corte_ganancia_maxima_validacion"] == bo["corte_validacion"])
            control = " | reproduce la bayesiana: " + ("sí" if igual else "NO (revisar)")
        print(f"  semilla {semilla}: ganancia 201908 ${metricas['ganancia_maxima_validacion'] / 1e6:,.1f}M en "
              f"{metricas['corte_ganancia_maxima_validacion']:,} | AUC {metricas['auc_validacion']:.4f} | "
              f"{metricas['segundos']:.0f} s{control}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", default=METODOS, choices=METODOS)
    parser.add_argument("--semillas", nargs="+", type=int, default=SEMILLAS)
    parser.add_argument("--forzar", action="store_true", help="rehace las semillas que ya tienen resultado")
    args = parser.parse_args()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    for metodo in args.metodos:
        validar_familia(metodo, args.semillas, args.forzar)


if __name__ == "__main__":
    main()
