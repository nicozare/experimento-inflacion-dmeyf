"""Rutas y constantes del experimento colaborativo de inflación (grupo Inflacion_2).

Todo lo que no es "cómo se corrigen las variables en pesos" sigue el
baseline que publicó Alejandro Bolaños para los experimentos colaborativos
(Zulip, noviembre 2026), con las aclaraciones que dio en el hilo:

  - Solo meses de 2019. 201911 (solo tiene BAJA+1) y 201912 (sin clase) son
    del dashboard: no se usan para nada.
  - Features: las originales + lag y delta de orden 1 y 2.
  - Target: BAJA+1 y BAJA+2 juntas.
  - Optimización bayesiana de num_leaves, learning_rate, min_data_in_leaf,
    feature_fraction y bagging_fraction; bagging_freq fijo >= 1; el resto
    de los hiperparámetros en el default de LightGBM.
  - Entrenamiento 201903-201906 completo (sin sampling), validación 201908
    con AUC (ordena la bayesiana y hace el early stopping).
  - Test 201910, medido con ganancia, solo para comparar. Para el test se
    reentrena con 201903-201908 ("Yes", respuesta de Ale en el hilo).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_DIR / "data"
RESULTADOS_DIR = REPO_DIR / "resultados"
OUTPUT_DIR = REPO_DIR / "output"

# Dataset crudo de la competencia 2 (2019-2021). Por defecto en data/; con la
# variable de entorno CRUDO se puede apuntar a otro archivo (.csv o .csv.gz).
RAW_PATH = Path(os.environ.get("CRUDO", DATA_DIR / "competencia_02_crudo.csv.gz"))

# Salidas intermedias: un parquet por mes.
BASE_DIR = DATA_DIR / "base"   # crudo de 2019 + clase_ternaria
FE_DIR = DATA_DIR / "fe"       # fe/<metodo>/<foto_mes>.parquet (corrección + lags/deltas)
OPTUNA_DB = DATA_DIR / "optuna.db"


def ruta_base(foto_mes: int) -> Path:
    return BASE_DIR / f"{foto_mes}.parquet"


def ruta_fe(metodo: str, foto_mes: int) -> Path:
    return FE_DIR / metodo / f"{foto_mes}.parquet"


# Datos extendidos (19/11, barrida de meses que pidió el tutor). Con
# DATOS=extendido, 01 y 02 arman 201901-202106 con la clase calculada con todo
# el crudo (hasta 202108), en data/extendido/: no tocan data/base ni data/fe
# del experimento principal. Los meses de 2019 salen idénticos (control en 14).
DATOS = os.environ.get("DATOS", "2019")
if DATOS not in ("2019", "extendido"):
    raise SystemExit(f"DATOS={DATOS} no existe; opciones: 2019, extendido")
if DATOS == "extendido":
    BASE_DIR = DATA_DIR / "extendido" / "base"
    FE_DIR = DATA_DIR / "extendido" / "fe"


# ---------------------------------------------------------------------------
# Meses (baseline de Ale)
# ---------------------------------------------------------------------------
MES_LIMITE_DATOS = 202108 if DATOS == "extendido" else 201912  # la clase se calcula solo con datos de 2019

# Ventana de tiempo. "201910" es el baseline de Ale (por defecto). "201909"
# (seguimiento S5) corre el mismo protocolo un mes antes, para confirmar en
# otro mes lo que se vio en 201910: entrena 201903-201905, valida 201907,
# reentrena 201903-201907 y testea 201909. Respeta el mismo hueco de un mes
# (la clase del mes t se completa con los datos de t+2) y no puede empezar
# antes de 201903 sin perder los lags de orden 2. Sus resultados van a
# resultados/test_201909/ y output/test_201909/, y sus estudios de Optuna
# llevan el sufijo _201909, así no se mezclan con los de 201910.
# 01, 02 y 02b arman los datos de todos los meses: se corren sin VENTANA.
VENTANA = os.environ.get("VENTANA", "201910")
VENTANAS = {
    "201910": {"entrenamiento": [201903, 201904, 201905, 201906], "validacion": 201908,
               "final": [201903, 201904, 201905, 201906, 201907, 201908], "test": 201910},
    "201909": {"entrenamiento": [201903, 201904, 201905], "validacion": 201907,
               "final": [201903, 201904, 201905, 201906, 201907], "test": 201909},
}
if VENTANA not in VENTANAS:
    raise SystemExit(f"VENTANA={VENTANA} no existe; opciones: {', '.join(VENTANAS)}")
MESES_ENTRENAMIENTO = VENTANAS[VENTANA]["entrenamiento"]
MES_VALIDACION = VENTANAS[VENTANA]["validacion"]
MESES_ENTRENAMIENTO_FINAL = VENTANAS[VENTANA]["final"]
MES_TEST = VENTANAS[VENTANA]["test"]   # con 201910, el último mes que se procesa; 201911 y 201912 ni se tocan
SUFIJO_VENTANA = "" if VENTANA == "201910" else f"_{VENTANA}"
if SUFIJO_VENTANA:
    RESULTADOS_DIR = RESULTADOS_DIR / f"test_{VENTANA}"
    OUTPUT_DIR = OUTPUT_DIR / f"test_{VENTANA}"

# Barrida de meses: resultados y salidas aparte. Con DATOS=extendido, 01 guarda
# hasta 202106 (el último mes con la clase completa: BAJA+2 necesita los dos
# meses siguientes) y las salidas de 01 y 02 (clase por mes, celdas rotas,
# escalas) van a output/barrida/, así no pisan las del experimento principal.
RESULTADOS_BARRIDA = REPO_DIR / "resultados" / "barrida"
OUTPUT_BARRIDA = REPO_DIR / "output" / "barrida"
MES_ULTIMO_DATOS = 202106 if DATOS == "extendido" else MES_TEST
OUTPUT_DATOS = OUTPUT_BARRIDA if DATOS == "extendido" else OUTPUT_DIR


def exigir_ventana_principal(script: str) -> None:
    """01, 02 y 02b borran y rearman data/ para todos los meses hasta MES_TEST:
    con otra ventana borrarían 201910."""
    if SUFIJO_VENTANA:
        raise SystemExit(f"{script} arma los datos de todos los meses: correrlo sin VENTANA")

# ---------------------------------------------------------------------------
# Familias del experimento. La única diferencia entre ellas es cómo se
# corrigen las variables en pesos ANTES de calcular lags y deltas.
#   ninguna  F0: valores nominales (el baseline de Ale tal cual)
#   ipc      F1: deflactadas por IPC (referencia externa, punto en común con Inflacion_1)
#   mediana  F2: divididas por la mediana del mes (deflactor endógeno)
#   ranking  F3: posición del cliente dentro del mes, con el cero fijo
# ---------------------------------------------------------------------------
METODOS_CORRECCION = ["ninguna", "ipc", "mediana", "ranking"]

# Experimento de seguimiento (18/11), decidido después de ver los resultados de F0-F3:
#   proporciones  F4: F0 + proporciones entre montos del mismo cliente y mes
#                 (scripts/02b_proporciones.py). No corrige: AGREGA columnas
#                 en las que la inflación se cancela sola (numerador y
#                 denominador están en pesos del mismo mes).
#   diferencias   F5: control de F4 (seguimiento S8, 18/11 a la tarde). Los
#                 mismos pares de montos, restados en vez de divididos, y nulos
#                 en las mismas celdas que F4. La diferencia sí arrastra la
#                 inflación (y el tamaño del cliente): si F5 rinde como F4, la
#                 mejora de F4 viene de la información nueva y no de que el
#                 cociente no tenga inflación.
METODOS = METODOS_CORRECCION + ["proporciones", "diferencias"]
NOMBRE_FAMILIA = {"ninguna": "F0 nominal", "ipc": "F1 IPC", "mediana": "F2 mediana", "ranking": "F3 ranking",
                  "proporciones": "F4 proporciones", "diferencias": "F5 diferencias"}

# F4: nombre -> (columnas que se suman arriba, columnas que se suman abajo).
# Si el denominador es 0 o nulo, la proporción queda nula. Los nulos de cada
# suma cuentan como 0 (salvo que todos sean nulos).
SUELDO = ["mpayroll", "mpayroll2"]
PROPORCIONES = {
    # el sueldo como escala del propio cliente
    "prop_saldo_sueldo": (["mcuentas_saldo"], SUELDO),
    "prop_consumo_tarjetas_sueldo": (["mtarjeta_visa_consumo", "mtarjeta_master_consumo"], SUELDO),
    "prop_prestamos_sueldo": (["mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios"], SUELDO),
    # (sin mttarjeta_visa_debitos_automaticos: celda rota en 201904)
    "prop_debitos_sueldo": (["mcuenta_debitos_automaticos", "mttarjeta_master_debitos_automaticos",
                             "mpagodeservicios", "mpagomiscuentas"], SUELDO),
    "prop_extracciones_sueldo": (["mextraccion_autoservicio", "matm", "matm_other"], SUELDO),
    # tarjetas: uso del límite, cuánto del saldo se paga y peso del consumo en dólares
    "prop_visa_uso_limite": (["Visa_msaldototal"], ["Visa_mlimitecompra"]),
    "prop_master_uso_limite": (["Master_msaldototal"], ["Master_mlimitecompra"]),
    "prop_visa_pago_saldo": (["Visa_mpagospesos"], ["Visa_msaldopesos"]),
    "prop_master_pago_saldo": (["Master_mpagospesos"], ["Master_msaldopesos"]),
    "prop_visa_consumo_dolares": (["Visa_mconsumosdolares"], ["Visa_mconsumototal"]),
    # composición del patrimonio. (Transferencias emitidas / recibidas quedó
    # afuera: mtransferencias_recibidas está rota de 201901 a 201905.)
    "prop_dolarizacion": (["mcaja_ahorro_dolares", "mplazo_fijo_dolares", "minversion1_dolares"],
                          ["mcaja_ahorro", "mcaja_ahorro_adicional", "mcaja_ahorro_dolares", "mplazo_fijo_pesos",
                           "mplazo_fijo_dolares", "minversion1_pesos", "minversion1_dolares", "minversion2"]),
}

# Variables en pesos: las 73 que el diccionario de datos marca con unidad
# "pesos", que son exactamente las que empiezan con m, Visa_m o Master_m
# (misma convención que usa la cátedra). Incluye las que nacen en dólares
# pero el diccionario dice que vienen "expresadas en pesos" al tipo de
# cambio de cierre: también suben, pero con el dólar (ver README).
# Cantidades, fechas, días, edad y flags no se tocan.
REGEX_MONETARIAS = r"^(m|Visa_m|Master_m)"

# IPC Nacional, nivel general, base diciembre 2016 = 100 (INDEC). Serie
# 148.3_INIVELNAL_DICI_M_26 de la API de series de tiempo de datos.gob.ar
# (2019 cargado el 17/11; 2020-2021 agregado el 19/11 para la barrida de meses).
# F1 lleva todo a pesos de MES_REFERENCIA_IPC (x * IPC[ref] / IPC[mes]).
# Multiplicar todo por una constante no cambia los árboles, así que el mes
# de referencia no afecta el resultado: solo cómo se leen los valores.
MES_REFERENCIA_IPC = 201910
IPC = {
    201901: 189.6101, 201902: 196.7501, 201903: 205.9571, 201904: 213.0517,
    201905: 219.5691, 201906: 225.5370, 201907: 230.4940, 201908: 239.6077,
    201909: 253.7102, 201910: 262.0661, 201911: 273.2158, 201912: 283.4442,
    202001: 289.8299, 202002: 295.6660, 202003: 305.5515, 202004: 310.1243,
    202005: 314.9087, 202006: 321.9738, 202007: 328.2014, 202008: 337.0632,
    202009: 346.6207, 202010: 359.6570, 202011: 371.0211, 202012: 385.8826,
    202101: 401.5071, 202102: 415.8595, 202103: 435.8657, 202104: 453.6503,
    202105: 468.7250, 202106: 483.6049, 202107: 498.0987, 202108: 510.3942,
}

# Celdas rotas (una variable en 0 para casi todos los clientes en un mes
# puntual, p. ej. mcomisiones en 201905 y 201910). El baseline no las
# corrige, así que por defecto quedan como están; 01 igual las lista en
# output/celdas_rotas.csv porque son una limitación a mencionar.
CORREGIR_CELDAS_ROTAS = False
UMBRAL_CELDA_ROTA = 0.02

# ---------------------------------------------------------------------------
# Optimización bayesiana (Optuna, TPE). Mismo rango y mismo presupuesto para
# las 4 familias. Rangos elegidos por el grupo (la consigna lo deja libre).
# ---------------------------------------------------------------------------
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
SEMILLA_BO = 100081
ESPACIO_BUSQUEDA = {
    "num_leaves": (8, 1024),            # entero, escala log
    "learning_rate": (0.005, 0.3),      # escala log
    "min_data_in_leaf": (10, 8000),     # entero, escala log
    "feature_fraction": (0.1, 1.0),
    "bagging_fraction": (0.2, 1.0),
}
BAGGING_FREQ = 1
MAX_RONDAS = 5000
EARLY_STOPPING = 200

# Parámetros fijos. Todo lo que no figura acá queda en el default de LightGBM.
# deterministic y force_row_wise no cambian lo que el modelo aprende: fijan
# el orden de las sumas para que la misma semilla dé siempre el mismo modelo
# (si no, las diferencias entre familias podrían ser numéricas).
# feature_pre_filter=False es necesario para que la bayesiana pueda cambiar
# min_data_in_leaf sin rearmar el dataset en cada intento.
PARAMS_FIJOS = {
    "objective": "binary",
    "metric": "auc",
    "bagging_freq": BAGGING_FREQ,
    "feature_pre_filter": False,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
    "num_threads": int(os.environ.get("HILOS", "0")),
}

# ---------------------------------------------------------------------------
# Modelo final y test
# ---------------------------------------------------------------------------
# 20 semillas por familia: las 5 asignadas por la cátedra a Nicolás y los 15
# primos siguientes. Las mismas 20 en las 4 familias, así la comparación es
# pareada por semilla (Wilcoxon). Se pueden pisar con SEMILLAS="1,2,3".
SEMILLAS = [int(s) for s in os.environ.get("SEMILLAS", ",".join(map(str, [
    100081, 100103, 100129, 100151, 100169, 100183, 100189, 100193, 100207, 100213,
    100237, 100267, 100271, 100279, 100291, 100297, 100313, 100333, 100343, 100357,
]))).split(",")]

# Función de ganancia (la misma de la Primera Competencia).
GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500

# Barrido de la cantidad de clientes a estimular (curvas de ganancia).
CORTE_MINIMO, CORTE_MAXIMO, PASO_CORTE = 500, 20_000, 100

# ---------------------------------------------------------------------------
# Barrida de meses (19/11). El tutor: "que no mejore es un resultado válido,
# pero tiene que poder demostrarse en más de un mes". Usa DATOS=extendido.
# Las reglas de S9 y S10 están en el README, escritas antes de correr.
# ---------------------------------------------------------------------------
SEMILLAS_BARRIDA = SEMILLAS[:5]  # las 5 que la cátedra le asignó a Nicolás

# S9, el modelo que envejece (14): los modelos finales de 2019 (201903-201908,
# los mismos de 04) aplicados sin reentrenar a cada mes de 201910 a 202106.
MESES_ENVEJECE = [201910, 201911, 201912] + [a * 100 + m for a in (2020, 2021) for m in range(1, 13)
                                             if a * 100 + m <= 202106]

# S10, la barrida con ventana móvil (15): la ventana del baseline corrida en el
# tiempo. Para el mes de test t: entrena t-7 a t-4, elige el corte en t-2,
# reentrena t-7 a t-2. Con t = 201910 es exactamente la ventana principal.
MESES_TEST_BARRIDA = [201910, 202001, 202004, 202007, 202010, 202101, 202104]

# Hipótesis secundarias del tutor, como familias nuevas de S10. Usan los
# hiperparámetros de la bayesiana de 2019 de la familia de la que salen.
#   ninguna_sin_drift  F6: F0 sin las variables con mucho drift (PSI > 0,25)
#   mediana_sin_drift  F7: F2 sin las variables con mucho drift (PSI > 0,25,
#                      medido después de dividir por la mediana)
#   ranking_con_drift  F8: F3 + ranking de las variables NO monetarias con
#                      drift (PSI > 0,1); sale de data/extendido/fe/ranking_todo
# El PSI (el de 05) se mide entre los meses del reentreno y el mes de test, sin
# la clase: se puede calcular en producción. Umbrales de la convención usual:
# 0,1 = cambio moderado, 0,25 = cambio fuerte.
PSI_CON_DRIFT, PSI_MUCHO_DRIFT = 0.10, 0.25
FAMILIAS_DRIFT = {"ninguna_sin_drift": "ninguna", "mediana_sin_drift": "mediana", "ranking_con_drift": "ranking"}
NOMBRE_FAMILIA |= {"ninguna_sin_drift": "F6 sin las de mucho drift",
                   "mediana_sin_drift": "F7 mediana sin las de mucho drift",
                   "ranking_con_drift": "F8 ranking de todo lo que drifta"}
