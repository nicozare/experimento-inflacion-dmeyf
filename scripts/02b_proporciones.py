"""F4 (experimento de seguimiento): agrega proporciones al dataset nominal.

Tercera forma de neutralizar la inflación, sin índices externos (lo que
estudia Inflacion_1) ni la distribución del mes (F2 y F3): dividir un monto
del cliente por otro monto del MISMO cliente y mes. La inflación mueve igual
al numerador y al denominador, y se cancela. Ejemplos: meses de sueldo que
tiene en la cuenta, qué parte del límite de la tarjeta usa, cuánto del saldo
paga, qué parte del patrimonio está en dólares (lista en config.PROPORCIONES).

A diferencia de F1-F3, F4 no reemplaza nada: es F0 (nominal) con 11
proporciones más, cada una con su lag y delta de orden 1 y 2 (55 columnas).
Mide otra cosa: si sumar información libre de inflación ayuda.

Se arma desde data/fe/ninguna (02 tiene que haber corrido antes): las
proporciones de los meses anteriores salen de los lags que ya están ahí
(lag_k de la proporción = lag_k del numerador / lag_k del denominador), así
que usan exactamente la misma definición de lag que el resto.

Salida:
  data/fe/proporciones/<foto_mes>.parquet
  output/proporciones_psi.csv   cobertura y PSI (201903-201908 contra 201910)
                                de cada proporción y de su numerador nominal

Uso:
    uv run python scripts/02b_proporciones.py
"""
from __future__ import annotations

import importlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from comun import meses_disponibles, vaciar_carpeta
from config import FE_DIR, MES_TEST, MESES_ENTRENAMIENTO_FINAL, OUTPUT_DIR, PROPORCIONES, ruta_fe

psi = importlib.import_module("05_drift").psi  # misma definición de PSI que el resto del experimento


def suma(tabla: pa.Table, columnas: list[str], prefijo: str = "") -> np.ndarray:
    """Suma de columnas; los nulos cuentan como 0 salvo que todos sean nulos."""
    v = np.column_stack([tabla.column(prefijo + c).to_numpy(zero_copy_only=False).astype(np.float64)
                         for c in columnas])
    total = np.nansum(v, axis=1)
    total[np.isnan(v).all(axis=1)] = np.nan
    return total


def proporcion(tabla: pa.Table, arriba: list[str], abajo: list[str], prefijo: str = "") -> np.ndarray:
    num, den = suma(tabla, arriba, prefijo), suma(tabla, abajo, prefijo)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r[~np.isfinite(r)] = np.nan  # denominador 0 o nulo
    return r


def main() -> None:
    t0 = time.time()
    origen = FE_DIR / "ninguna"
    meses = meses_disponibles(origen)
    if not meses:
        raise SystemExit("Falta data/fe/ninguna: correr antes scripts/02_corregir_y_fe.py")
    vaciar_carpeta(ruta_fe("proporciones", meses[0]).parent)
    print(f"=== proporciones: {len(PROPORCIONES)} proporciones x 5 = {5 * len(PROPORCIONES)} columnas nuevas, "
          f"meses {meses[0]}-{meses[-1]} ===", flush=True)

    valores = {nombre: {} for nombre in PROPORCIONES}  # para el PSI: nivel de cada proporción por mes
    numeradores = {nombre: {} for nombre in PROPORCIONES}
    for mes in meses:
        tabla = pq.read_table(ruta_fe("ninguna", mes))
        niveles, extras = [], []
        for nombre, (arriba, abajo) in PROPORCIONES.items():
            r0 = proporcion(tabla, arriba, abajo)
            r1 = proporcion(tabla, arriba, abajo, "lag_1_")
            r2 = proporcion(tabla, arriba, abajo, "lag_2_")
            niveles.append((nombre, r0))
            extras += [(f"lag_1_{nombre}", r1), (f"lag_2_{nombre}", r2),
                       (f"delta_1_{nombre}", r0 - r1), (f"delta_2_{nombre}", r0 - r2)]
            valores[nombre][mes] = r0
            numeradores[nombre][mes] = suma(tabla, arriba)
        for col, v in niveles + extras:
            tabla = tabla.append_column(col, pa.array(v.astype(np.float32)))
        pq.write_table(tabla, ruta_fe("proporciones", mes))
        print(f"  {mes}: {tabla.num_rows:,} filas, {tabla.num_columns - 3} features", flush=True)

    filas = []
    for nombre, (arriba, abajo) in PROPORCIONES.items():
        ref = np.concatenate([valores[nombre][m] for m in MESES_ENTRENAMIENTO_FINAL])
        test = valores[nombre][MES_TEST]
        ref_num = np.concatenate([numeradores[nombre][m] for m in MESES_ENTRENAMIENTO_FINAL])
        filas.append({"proporcion": nombre, "numerador": " + ".join(arriba), "denominador": " + ".join(abajo),
                      "pct_con_dato_201910": round(100 * np.mean(~np.isnan(test)), 1),
                      "psi_proporcion": round(psi(ref, test), 4),
                      "psi_numerador_nominal": round(psi(ref_num, numeradores[nombre][MES_TEST]), 4)})
    tabla_psi = pd.DataFrame(filas)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla_psi.to_csv(OUTPUT_DIR / "proporciones_psi.csv", index=False)
    print("\nCobertura y PSI (201903-201908 contra 201910):")
    print(tabla_psi[["proporcion", "pct_con_dato_201910", "psi_proporcion", "psi_numerador_nominal"]]
          .to_string(index=False))
    print(f"\nproporciones listo ({time.time() - t0:.0f} s) -> {origen.parent / 'proporciones'}")


if __name__ == "__main__":
    main()
