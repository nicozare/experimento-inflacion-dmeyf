"""Funciones compartidas por los scripts del experimento."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from config import (
    BASE_DIR,
    CORTE_MAXIMO,
    CORTE_MINIMO,
    COSTO_ESTIMULO,
    GANANCIA_ACIERTO,
    PASO_CORTE,
    REGEX_MONETARIAS,
    RESULTADOS_DIR,
    ruta_fe,
)

COLUMNAS_CLAVE = ["numero_de_cliente", "foto_mes", "clase_ternaria"]


# ---------------------------------------------------------------------------
# Meses, columnas y carpetas
# ---------------------------------------------------------------------------
def meses_disponibles(directorio: Path = BASE_DIR) -> list[int]:
    """Meses (foto_mes) que tienen su parquet en `directorio`."""
    return sorted(int(p.stem) for p in directorio.glob("*.parquet"))


def es_monetaria(columna: str) -> bool:
    return re.match(REGEX_MONETARIAS, columna) is not None


def columnas_features(metodo: str, mes: int) -> list[str]:
    """Columnas de entrada del modelo: todo menos las claves y la clase."""
    esquema = pq.read_schema(ruta_fe(metodo, mes))
    return [c for c in esquema.names if c not in COLUMNAS_CLAVE]


def vaciar_carpeta(carpeta: Path) -> None:
    """Borra y recrea una carpeta de salidas regenerables (data/...)."""
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lectura del dataset de features
# ---------------------------------------------------------------------------
def cargar_meses(metodo: str, meses: list[int], feats: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Matriz float32 (filas x features) con los meses pedidos, uno debajo del
    otro, y sus claves (cliente, mes, clase) en el mismo orden de filas. Los
    nulos quedan como NaN (LightGBM los trata como faltantes)."""
    claves = [pq.read_table(ruta_fe(metodo, m), columns=COLUMNAS_CLAVE).to_pandas() for m in meses]
    n_total = sum(len(c) for c in claves)
    X = np.empty((n_total, len(feats)), dtype=np.float32)
    fila = 0
    for mes, c in zip(meses, claves):
        tabla = pq.read_table(ruta_fe(metodo, mes), columns=feats)
        for j, columna in enumerate(tabla.columns):
            X[fila:fila + len(c), j] = columna.to_numpy()
        fila += len(c)
        del tabla
    return X, pd.concat(claves, ignore_index=True)


def etiqueta_binaria(clase_ternaria: pd.Series) -> np.ndarray:
    """BAJA+1 y BAJA+2 juntas como positivo (baseline de Ale)."""
    return clase_ternaria.isin(["BAJA+1", "BAJA+2"]).to_numpy().astype(np.int8)


# ---------------------------------------------------------------------------
# Ganancia
# ---------------------------------------------------------------------------
def cortes_barrido() -> np.ndarray:
    return np.arange(CORTE_MINIMO, CORTE_MAXIMO + 1, PASO_CORTE)


def ganancia_por_corte(prob: np.ndarray, clase_ternaria: np.ndarray, cortes: np.ndarray) -> np.ndarray:
    """Ganancia de estimular a los `corte` clientes de mayor probabilidad,
    para cada corte. Solo BAJA+2 suma; cualquier otro estímulo resta."""
    orden = np.argsort(-prob, kind="stable")
    g = np.where(np.asarray(clase_ternaria)[orden] == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO).astype(np.int64)
    acumulada = np.cumsum(g)
    return acumulada[np.minimum(cortes, len(acumulada)) - 1]


def auc(y: np.ndarray, prob: np.ndarray) -> float:
    """AUC por rangos (Mann-Whitney), con empates promediados."""
    r = rankdata(prob)
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    return float((r[np.asarray(y) == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Resultados de la bayesiana (se comparten entre integrantes vía git)
# ---------------------------------------------------------------------------
def ruta_bo(metodo: str) -> Path:
    return RESULTADOS_DIR / f"bo__{metodo}.json"


def leer_bo(metodo: str) -> dict:
    ruta = ruta_bo(metodo)
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}: correr antes scripts/03_optimizar.py --metodos {metodo}")
    return json.loads(ruta.read_text(encoding="utf-8"))
