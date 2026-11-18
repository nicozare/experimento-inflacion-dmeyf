"""Aplica la corrección de inflación y arma lags y deltas, para cada familia.

Las 4 familias (config.METODOS_CORRECCION) difieren SOLO en cómo se corrigen las 73
variables en pesos antes de los lags/deltas. El valor corregido reemplaza al
original (misma columna, mismo nombre): no se agregan columnas, así la única
diferencia con el baseline es la corrección. Cantidades, fechas, días, edad
y flags quedan igual. La escala usada (mediana, factor de IPC) es solo un
número para hacer la cuenta: el modelo nunca la ve.

  ninguna  F0. Sin corregir (el baseline de Ale).
  ipc      F1. Cada monto se lleva a pesos de config.MES_REFERENCIA_IPC con el
           IPC del INDEC: x * IPC[ref] / IPC[mes].
  mediana  F2. En cada mes, cada variable se divide por la mediana de |x| entre
           los clientes con x distinto de 0 ("los que tienen el producto").
           Es una sola regla para todas las variables (muchas tienen mediana
           0), respeta el signo, y no se mueve si cambia la proporción de
           clientes sin el producto. Un cliente con el doble del monto típico
           vale 2 en cualquier mes.
  ranking  F3. En cada mes, el lugar del cliente entre los que tienen esa
           variable con el mismo signo ("rank con cero fijo"): positivos en
           (0, 1], negativos en [-1, 0), el 0 queda en 0 y el nulo queda nulo.
           1 = el que más tiene ese mes. Empates: comparten el valor.

Después de corregir: lags y deltas de orden 1 y 2 sobre TODAS las columnas
numéricas (corregidas o no): lag_k es el valor de la k-ésima fila anterior
del mismo cliente (como shift() del data.table de la cátedra o lag() de SQL;
si el cliente tuvo un hueco, no es necesariamente k meses atrás) y
delta_k = valor actual - lag_k.

Se recorre de a un mes, guardando solo las dos últimas filas de cada
cliente. Validado contra el dataset de features del candidato de la
competencia 1 (mismos valores celda por celda en 202106).

Uso:
    uv run python scripts/02_corregir_y_fe.py                  # las 4 familias
    uv run python scripts/02_corregir_y_fe.py --metodos mediana
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from comun import COLUMNAS_CLAVE, es_monetaria, meses_disponibles, vaciar_carpeta
from config import (IPC, MES_REFERENCIA_IPC, METODOS_CORRECCION, OUTPUT_DIR, exigir_ventana_principal,
                    ruta_base, ruta_fe)


def cume_dist(v: np.ndarray) -> np.ndarray:
    """Fracción de valores <= v (los empates comparten valor): en (0, 1]."""
    ordenado = np.sort(v)
    return np.searchsorted(ordenado, v, side="right") / len(v)


def corregir_mes(X: np.ndarray, mes: int, metodo: str, idx_monetarias: list[int]) -> tuple[np.ndarray, dict]:
    """Corrige, en el lugar, las columnas monetarias de X (filas = clientes de
    un mes). Devuelve X y la escala usada por columna (para auditar)."""
    escalas: dict = {}
    if metodo == "ninguna":
        return X, escalas
    if metodo == "ipc":
        factor = IPC[MES_REFERENCIA_IPC] / IPC[mes]
        X[:, idx_monetarias] *= factor
        return X, {"factor_ipc": factor}
    for j in idx_monetarias:
        x = X[:, j]
        if metodo == "mediana":
            no_cero = x[(x != 0) & ~np.isnan(x)]
            escala = float(np.median(np.abs(no_cero))) if len(no_cero) else np.nan
            escalas[j] = escala
            X[:, j] = np.where(x == 0, 0.0, x / escala)
        elif metodo == "ranking":
            r = np.full_like(x, np.nan)
            r[x == 0] = 0.0
            pos, neg = x > 0, x < 0
            if pos.any():
                r[pos] = cume_dist(x[pos])
            if neg.any():
                r[neg] = -cume_dist(-x[neg])
            X[:, j] = r
        else:
            raise ValueError(f"método desconocido: {metodo}")
    return X, escalas


def procesar_metodo(metodo: str, meses: list[int]) -> None:
    t0 = time.time()
    esquema = pq.read_schema(ruta_base(meses[0]))
    numericas = [c for c in esquema.names if c not in COLUMNAS_CLAVE]
    idx_monetarias = [j for j, c in enumerate(numericas) if es_monetaria(c)]
    p = len(numericas)
    print(f"\n=== {metodo}: {p} columnas numéricas, {len(idx_monetarias)} en pesos ===", flush=True)

    # universo de clientes -> índice denso, para guardar sus dos últimas filas
    clientes = np.unique(np.concatenate([
        pq.read_table(ruta_base(m), columns=["numero_de_cliente"]).column(0).to_numpy() for m in meses
    ]))
    ultima = np.full((len(clientes), p), np.nan)      # fila anterior del cliente
    anteultima = np.full((len(clientes), p), np.nan)  # la de antes

    # salida: las numéricas y después lag_1, lag_2, delta_1, delta_2 intercalados por columna
    nombres_lag = []
    for c in numericas:
        nombres_lag += [f"lag_1_{c}", f"lag_2_{c}", f"delta_1_{c}", f"delta_2_{c}"]
    nombres = numericas + nombres_lag

    # Buffers reservados UNA vez (tamaño del mes más grande) y reusados en cada
    # mes: pedir arreglos nuevos de ~150 MB por mes hacía que el allocator de
    # macOS retuviera ~1,3 GB por mes sin devolverlos.
    n_max = max(pq.read_metadata(ruta_base(m)).num_rows for m in meses)
    X_buf, lag1_buf, lag2_buf, d1_buf, d2_buf = (np.empty((n_max, p)) for _ in range(5))
    salida_buf = np.empty((n_max, len(nombres)), dtype=np.float32, order="F")  # columnas contiguas

    auditoria = []
    vaciar_carpeta(ruta_fe(metodo, meses[0]).parent)
    for mes in meses:
        tabla = pq.read_table(ruta_base(mes))
        n = tabla.num_rows
        idx = np.searchsorted(clientes, tabla.column("numero_de_cliente").to_numpy())
        X, lag1, lag2, d1, d2 = X_buf[:n], lag1_buf[:n], lag2_buf[:n], d1_buf[:n], d2_buf[:n]
        for j, c in enumerate(numericas):
            X[:, j] = tabla.column(c).to_numpy().astype(np.float64, copy=False)
        X, escalas = corregir_mes(X, mes, metodo, idx_monetarias)

        np.take(ultima, idx, axis=0, out=lag1, mode="clip")
        np.take(anteultima, idx, axis=0, out=lag2, mode="clip")
        anteultima[idx] = lag1
        ultima[idx] = X
        np.subtract(X, lag1, out=d1)
        np.subtract(X, lag2, out=d2)

        salida = salida_buf[:n]
        salida[:, :p] = X
        for k, arr in enumerate((lag1, lag2, d1, d2)):
            salida[:, p + k::4] = arr
        columnas = [tabla.column(c) for c in COLUMNAS_CLAVE] + [pa.array(salida[:, j]) for j in range(len(nombres))]
        pq.write_table(pa.Table.from_arrays(columnas, names=COLUMNAS_CLAVE + nombres), ruta_fe(metodo, mes))
        del columnas, tabla

        if metodo == "mediana":
            auditoria.append({"foto_mes": mes, **{numericas[j]: v for j, v in escalas.items()}})
        elif metodo == "ipc":
            auditoria.append({"foto_mes": mes, **escalas})
        print(f"  {mes}: {n:,} filas", flush=True)

    if auditoria:
        destino = OUTPUT_DIR / f"escalas_{metodo}.csv"
        pd.DataFrame(auditoria).to_csv(destino, index=False)
        print(f"  escalas usadas -> {destino}", flush=True)
    print(f"  {metodo} listo: {len(nombres)} features ({time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    exigir_ventana_principal("02_corregir_y_fe.py")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metodos", nargs="+", default=METODOS_CORRECCION, choices=METODOS_CORRECCION)
    args = parser.parse_args()
    meses = meses_disponibles()
    if not meses:
        raise SystemExit("No hay data/base/*.parquet: correr antes scripts/01_preparar_base.py")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for metodo in args.metodos:
        procesar_metodo(metodo, meses)


if __name__ == "__main__":
    main()
