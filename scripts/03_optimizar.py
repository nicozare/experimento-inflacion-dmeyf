"""Optimización bayesiana de LightGBM, una por familia (baseline de Ale).

Para cada familia: entrena con 201903-201906 completos (sin sampling),
BAJA+1 y BAJA+2 como positivo, y mide AUC en 201908 con early stopping. La
bayesiana (Optuna, TPE) maximiza ese AUC sobre num_leaves, learning_rate,
min_data_in_leaf, feature_fraction y bagging_fraction, con bagging_freq fijo
y el resto de los hiperparámetros en el default de LightGBM. Mismo rango,
misma semilla y mismo presupuesto (N_TRIALS) en las 4 familias.

201910 (test) no se toca acá.

De cada intento se guarda además la ganancia en 201908 y el corte (cantidad
de clientes) que la maximiza: el corte del mejor intento es el que después
se aplica en 201910, para no elegir el corte mirando el test.

Salida:
  resultados/bo__<metodo>.json         mejores hiperparámetros, iteraciones, AUC y corte
  resultados/bo__<metodo>__intentos.csv todos los intentos
  data/optuna.db                       estudio de Optuna (permite retomar si se corta)

Uso:
    uv run python scripts/03_optimizar.py                       # las 4 familias
    uv run python scripts/03_optimizar.py --metodos ranking --trials 50
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import lightgbm as lgb
import numpy as np
import optuna

from comun import (
    cargar_meses,
    columnas_features,
    cortes_barrido,
    etiqueta_binaria,
    ganancia_por_corte,
    ruta_bo,
)
from config import (
    EARLY_STOPPING,
    ESPACIO_BUSQUEDA,
    MAX_RONDAS,
    MES_VALIDACION,
    MESES_ENTRENAMIENTO,
    METODOS,
    N_TRIALS,
    OPTUNA_DB,
    PARAMS_FIJOS,
    RESULTADOS_DIR,
    SUFIJO_VENTANA,
    SEMILLA_BO,
)


def proponer(trial: optuna.Trial) -> dict:
    e = ESPACIO_BUSQUEDA
    return {
        "num_leaves": trial.suggest_int("num_leaves", *e["num_leaves"], log=True),
        "learning_rate": trial.suggest_float("learning_rate", *e["learning_rate"], log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *e["min_data_in_leaf"], log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", *e["feature_fraction"]),
        "bagging_fraction": trial.suggest_float("bagging_fraction", *e["bagging_fraction"]),
    }


def optimizar(metodo: str, n_trials: int) -> None:
    t0 = time.time()
    # si la familia ya tiene su bayesiana completa, no se cargan los datos ni se reescribe el json
    if ruta_bo(metodo).exists():
        hechos = json.loads(ruta_bo(metodo).read_text(encoding="utf-8"))["intentos_completos"]
        if hechos >= n_trials:
            print(f"\n=== {metodo}: la bayesiana ya está completa ({hechos} intentos, {ruta_bo(metodo)}) ===",
                  flush=True)
            return
    feats = columnas_features(metodo, MES_VALIDACION)
    X_train, k_train = cargar_meses(metodo, MESES_ENTRENAMIENTO, feats)
    X_valid, k_valid = cargar_meses(metodo, [MES_VALIDACION], feats)
    clase_valid = k_valid["clase_ternaria"].to_numpy()
    cortes = cortes_barrido()
    print(f"\n=== {metodo}: entrena {MESES_ENTRENAMIENTO} ({len(X_train):,} filas), "
          f"valida {MES_VALIDACION} ({len(X_valid):,} filas), {len(feats)} features ===", flush=True)

    # el dataset se binariza una vez y se reusa en todos los intentos
    params_dataset = {k: PARAMS_FIJOS[k] for k in ("feature_pre_filter", "verbosity", "num_threads")}
    dtrain = lgb.Dataset(X_train, label=etiqueta_binaria(k_train["clase_ternaria"]),
                         feature_name=feats, params=params_dataset, free_raw_data=False).construct()
    dvalid = lgb.Dataset(X_valid, label=etiqueta_binaria(k_valid["clase_ternaria"]),
                         reference=dtrain, params=params_dataset, free_raw_data=False).construct()

    def objetivo(trial: optuna.Trial) -> float:
        params = {**PARAMS_FIJOS, "seed": SEMILLA_BO, **proponer(trial)}
        inicio = time.time()
        modelo = lgb.train(params, dtrain, num_boost_round=MAX_RONDAS, valid_sets=[dvalid],
                           valid_names=["validacion"], callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        auc = modelo.best_score["validacion"]["auc"]
        prob = modelo.predict(X_valid, num_iteration=modelo.best_iteration)
        g = ganancia_por_corte(prob, clase_valid, cortes)
        trial.set_user_attr("best_iteration", int(modelo.best_iteration))
        trial.set_user_attr("corte_validacion", int(cortes[np.argmax(g)]))
        trial.set_user_attr("ganancia_validacion", int(g.max()))
        trial.set_user_attr("segundos", round(time.time() - inicio, 1))
        print(f"  intento {trial.number:3d}: AUC {auc:.5f} | {modelo.best_iteration:4d} árboles | "
              f"ganancia {MES_VALIDACION} ${g.max() / 1e6:,.1f}M en {cortes[np.argmax(g)]:,} | {time.time() - inicio:.0f} s",
              flush=True)
        return auc

    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    estudio = optuna.create_study(
        study_name=f"bo_{metodo}{SUFIJO_VENTANA}", storage=f"sqlite:///{OPTUNA_DB.as_posix()}", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA_BO),
    )
    hechos = len([t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE])
    faltan = max(0, n_trials - hechos)
    print(f"  intentos ya hechos: {hechos}, faltan: {faltan}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if faltan:
        estudio.optimize(objetivo, n_trials=faltan)

    mejor = estudio.best_trial
    resumen = {
        "metodo": metodo,
        "hiperparametros": mejor.params,
        "best_iteration": mejor.user_attrs["best_iteration"],
        "auc_validacion": mejor.value,
        "corte_validacion": mejor.user_attrs["corte_validacion"],
        "ganancia_validacion": mejor.user_attrs["ganancia_validacion"],
        "intento": mejor.number,
        "intentos_completos": len([t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "params_fijos": PARAMS_FIJOS,
        "semilla_bo": SEMILLA_BO,
        "meses_entrenamiento": MESES_ENTRENAMIENTO,
        "mes_validacion": MES_VALIDACION,
        "lightgbm": lgb.__version__,
        "maquina": f"{platform.system()} {platform.machine()} ({platform.processor() or 'sin dato'})",
        "segundos_esta_corrida": round(time.time() - t0),
    }
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_bo(metodo).write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    intentos = estudio.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    intentos.to_csv(RESULTADOS_DIR / f"bo__{metodo}__intentos.csv", index=False)
    print(f"  mejor intento {mejor.number}: AUC {mejor.value:.5f}, {resumen['best_iteration']} árboles, "
          f"corte {resumen['corte_validacion']:,} -> {ruta_bo(metodo)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", default=METODOS, choices=METODOS)
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="presupuesto total de intentos por familia")
    args = parser.parse_args()
    for metodo in args.metodos:
        optimizar(metodo, args.trials)


if __name__ == "__main__":
    main()
