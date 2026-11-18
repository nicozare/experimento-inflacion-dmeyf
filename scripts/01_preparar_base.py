"""Prepara la base común a las 4 familias del experimento.

Pasos (los mismos para todas las familias; la corrección de inflación
recién se aplica en 02):
  1. Se lee el crudo completo y se queda solo con 2019 (baseline de Ale).
  2. clase_ternaria: misma consulta que candidatos/competencia_01, porteo de
     dmeyf2026/monday/z101_target_sql. Calculada solo con 2019, 201911 queda
     con BAJA+1 únicamente y 201912 sin clase, como dice la consigna.
  3. Celdas rotas: (mes, variable) donde la variable vale 0 para casi todos
     los clientes en un mes puntual. El baseline no las corrige; se listan
     en output/celdas_rotas.csv (son una limitación: 201910, el mes de test,
     tiene 14). Con config.CORREGIR_CELDAS_ROTAS=True se pasan a NULL.
  4. Se guarda un parquet por mes, ordenado por cliente, SOLO hasta el mes de
     test (201910): 201911 y 201912 son del dashboard y no se usan.

Salida: data/base/<foto_mes>.parquet, output/clase_ternaria_por_mes.csv,
output/celdas_rotas.csv.
"""
from __future__ import annotations

import time

import duckdb
import pandas as pd

from comun import vaciar_carpeta
from config import (
    BASE_DIR,
    CORREGIR_CELDAS_ROTAS,
    DATA_DIR,
    MES_LIMITE_DATOS,
    MES_TEST,
    OUTPUT_DIR,
    RAW_PATH,
    UMBRAL_CELDA_ROTA,
    ruta_base,
)

NO_NUMERICAS = ("numero_de_cliente", "foto_mes", "clase_ternaria")
TIPOS_NUMERICOS = ("BIGINT", "DOUBLE", "INTEGER", "FLOAT", "HUGEINT", "SMALLINT", "TINYINT", "DECIMAL")


def detectar_celdas_rotas(con: duckdb.DuckDBPyConnection, numericas: list[str]) -> pd.DataFrame:
    """Una celda (mes, variable) está rota si su fracción de valores distintos
    de 0 es menor a UMBRAL_CELDA_ROTA veces la fracción típica (mediana entre
    meses) de esa variable. Solo mira los meses que se usan (hasta MES_TEST)."""
    fracciones = ", ".join(
        f'avg(CASE WHEN "{c}" IS NOT NULL AND "{c}" <> 0 THEN 1.0 ELSE 0.0 END) AS "{c}"' for c in numericas
    )
    frac = con.execute(
        f"select foto_mes, {fracciones} from base where foto_mes <= {MES_TEST} group by 1 order by 1"
    ).df().set_index("foto_mes")
    filas = []
    for c in numericas:
        tipica = frac[c].median()
        if tipica < 0.01:  # variables casi siempre en cero: no se puede distinguir un mes roto
            continue
        for mes, f in frac[c].items():
            if f < UMBRAL_CELDA_ROTA * tipica:
                filas.append({"foto_mes": int(mes), "variable": c,
                              "fraccion_no_cero": round(float(f), 5), "fraccion_tipica": round(float(tipica), 5)})
    return pd.DataFrame(filas, columns=["foto_mes", "variable", "fraccion_no_cero", "fraccion_tipica"])


def main() -> None:
    t0 = time.time()
    tmp = DATA_DIR / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    print(f"Leyendo {RAW_PATH} (solo hasta {MES_LIMITE_DATOS}) ...", flush=True)
    con.execute(f"""
        create or replace table crudo as
        select * from read_csv_auto('{RAW_PATH.as_posix()}', sample_size=-1)
        where foto_mes <= {MES_LIMITE_DATOS}
    """)
    tipos = {r[0]: r[1] for r in con.execute("describe crudo").fetchall()}
    numericas = [c for c, t in tipos.items() if c not in NO_NUMERICAS and t.split("(")[0] in TIPOS_NUMERICOS]
    print(f"  {con.execute('select count(*) from crudo').fetchone()[0]:,} filas de 2019, "
          f"{len(numericas)} columnas numéricas", flush=True)

    # clase_ternaria (consulta del candidato de la competencia 1)
    con.execute("""
        create or replace table base as
        with periodos as (select distinct foto_mes from crudo),
        clientes as (select distinct numero_de_cliente from crudo),
        todo as (select numero_de_cliente, foto_mes from clientes cross join periodos),
        ct as (
            select c.*, if(c.numero_de_cliente is null, 0, 1) as mes_0,
            lead(mes_0, 1) over (partition by t.numero_de_cliente order by foto_mes) as mes_1,
            lead(mes_0, 2) over (partition by t.numero_de_cliente order by foto_mes) as mes_2,
            case when mes_1 is null then null when mes_1 = 0 then 'BAJA+1'
                 when mes_2 is null then null when mes_2 = 0 then 'BAJA+2' else 'CONTINUA' end as clase_ternaria
            from todo t left join crudo c using (numero_de_cliente, foto_mes)
        ) select * exclude (mes_0, mes_1, mes_2) from ct where mes_0 = 1
    """)
    con.execute("drop table crudo")
    clase = con.execute("""
        select foto_mes, coalesce(clase_ternaria, 'SIN CLASE') as clase_ternaria, count(*) as n
        from base group by all order by all
    """).df().pivot(index="foto_mes", columns="clase_ternaria", values="n").fillna(0).astype(int)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clase.to_csv(OUTPUT_DIR / "clase_ternaria_por_mes.csv")
    print(clase.to_string(), flush=True)

    rotas = detectar_celdas_rotas(con, numericas)
    rotas.to_csv(OUTPUT_DIR / "celdas_rotas.csv", index=False)
    print(f"\nCeldas rotas (hasta {MES_TEST}): {len(rotas)} (mes, variable)", flush=True)
    for mes, g in rotas.groupby("foto_mes"):
        print(f"  {mes}: {', '.join(g['variable'])}", flush=True)
    if CORREGIR_CELDAS_ROTAS and len(rotas):
        exprs = []
        for c in numericas:
            meses = rotas.loc[rotas["variable"] == c, "foto_mes"].tolist()
            if meses:
                exprs.append(f'CASE WHEN foto_mes IN ({", ".join(map(str, meses))}) THEN NULL ELSE "{c}" END AS "{c}"')
        con.execute(f"create or replace table base as select * replace ({', '.join(exprs)}) from base")
        print("  -> pasadas a NULL (CORREGIR_CELDAS_ROTAS=True)", flush=True)
    else:
        print("  -> se dejan como están (el baseline no las corrige)", flush=True)

    # un parquet por mes hasta MES_TEST, ordenado por cliente
    orden = ["numero_de_cliente", "foto_mes", "clase_ternaria"] + numericas
    lista = ", ".join(f'"{c}"' for c in orden)
    vaciar_carpeta(BASE_DIR)
    meses = [r[0] for r in con.execute(
        f"select distinct foto_mes from base where foto_mes <= {MES_TEST} order by 1").fetchall()]
    for mes in meses:
        con.execute(f"""
            copy (select {lista} from base where foto_mes = {mes} order by numero_de_cliente)
            to '{ruta_base(mes).as_posix()}' (format parquet)
        """)
    print(f"\n{len(meses)} meses ({meses[0]} a {meses[-1]}) guardados en {BASE_DIR} ({time.time() - t0:.0f} s)",
          flush=True)


if __name__ == "__main__":
    main()
