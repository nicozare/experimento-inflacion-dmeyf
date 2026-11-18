"""S8 (partes 1 y 2): por qué funciona F4. Sin regla de decisión.

1. Qué usa el modelo: la importancia (ganancia de los cortes) de las 55
   columnas nuevas, por proporción, en los modelos finales de F4 que guardó
   11 (data/modelos/). Cada variable junta su nivel, lags y deltas.
2. Dónde gana: con el mismo corte para las dos familias (4.000, el de F0),
   la ganancia de F4 menos la de F0 en 201910, separada entre los clientes
   que cobran el sueldo en el banco (donde existen las 5 proporciones contra
   el sueldo) y los que no. Con las 20 semillas de 04.

La parte 3 de S8 (F5, control con diferencias) es una familia más: sale de
02b, 03, 04, 07, 08 y 06.

Salida:
  output/por_que_f4.md
  output/importancia_f4.png

Uso:
    uv run python scripts/12_por_que_f4.py
"""
from __future__ import annotations

import importlib

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from comun import es_monetaria
from config import COSTO_ESTIMULO, GANANCIA_ACIERTO, MES_TEST, OUTPUT_DIR, PROPORCIONES, SEMILLAS, ruta_base

r6 = importlib.import_module("06_resumen")
r8 = importlib.import_module("08_corte_estable_y_ensamble")
r11 = importlib.import_module("11_estres_inflacion")  # dónde guarda los modelos
separar = importlib.import_module("09_donde_corta").separar
millones, miles, decimal, tabla_md = r6.millones, r6.miles, r6.decimal, r6.tabla_md
COLOR, FONDO, TINTA, TINTA_2 = r6.COLOR, r6.FONDO, r6.TINTA, r6.TINTA_2
CORTE = 4_000
NOMBRE_PROPORCION = {
    "prop_saldo_sueldo": "saldo / sueldo", "prop_consumo_tarjetas_sueldo": "consumo con tarjetas / sueldo",
    "prop_prestamos_sueldo": "préstamos / sueldo", "prop_debitos_sueldo": "débitos / sueldo",
    "prop_extracciones_sueldo": "extracciones / sueldo", "prop_visa_uso_limite": "uso del límite Visa",
    "prop_master_uso_limite": "uso del límite Master", "prop_visa_pago_saldo": "pago del saldo Visa",
    "prop_master_pago_saldo": "pago del saldo Master", "prop_visa_consumo_dolares": "consumo en dólares Visa",
    "prop_dolarizacion": "dolarización",
}


def importancia() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Parte de la ganancia de los cortes por variable (nivel + lags + deltas), media entre semillas."""
    rutas = [p for s in SEMILLAS if (p := r11.ruta_modelo("proporciones", s)).exists()]
    if not rutas:
        raise SystemExit("Faltan los modelos de F4 en data/modelos/: correr antes scripts/11_estres_inflacion.py")
    partes = []
    for ruta in rutas:
        m = lgb.Booster(model_file=str(ruta))
        imp = pd.Series(m.feature_importance(importance_type="gain"), index=m.feature_name())
        base = imp.groupby([separar(f)[0] for f in imp.index]).sum()
        partes.append(100 * base / base.sum())
    por_variable = pd.concat(partes, axis=1).mean(axis=1).sort_values(ascending=False)
    grupo = np.where(por_variable.index.str.startswith("prop_"), "proporciones (F4)",
                     np.where([es_monetaria(v) for v in por_variable.index], "montos en pesos",
                              "el resto (cantidades, flags, fechas, edad...)"))
    por_grupo = por_variable.groupby(grupo).sum().sort_values(ascending=False)
    props = pd.DataFrame({"proporción": [NOMBRE_PROPORCION[v] for v in PROPORCIONES],
                          "% de la ganancia de los cortes": [por_variable[v] for v in PROPORCIONES],
                          "puesto entre todas las variables": [int(np.nonzero(por_variable.index == v)[0][0]) + 1
                                                               for v in PROPORCIONES]})
    props = props.sort_values("% de la ganancia de los cortes", ascending=False)
    return props, por_grupo.rename_axis("grupo").reset_index(name="%"), len(por_variable)


def segmentos() -> pd.DataFrame:
    """Ganancia de F4 menos la de F0 con el mismo corte, por clientes con y sin sueldo."""
    clientes, p0 = r8.predicciones("prediccion", "ninguna", SEMILLAS)
    c4, p4 = r8.predicciones("prediccion", "proporciones", SEMILLAS)
    if not np.array_equal(clientes, c4):
        raise SystemExit("F0 y F4 no tienen los clientes en el mismo orden")
    base = pq.read_table(ruta_base(MES_TEST), columns=["numero_de_cliente", "mpayroll", "mpayroll2",
                                                        "clase_ternaria"]).to_pandas().set_index("numero_de_cliente")
    base = base.loc[clientes]
    con_sueldo = (base[["mpayroll", "mpayroll2"]].fillna(0).sum(axis=1) > 0).to_numpy()
    baja2 = (base.clase_ternaria == "BAJA+2").to_numpy()
    valor = np.where(baja2, GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    filas = []
    for s in SEMILLAS:
        for metodo, p in [("ninguna", p0[s]), ("proporciones", p4[s])]:
            elegidos = np.zeros(len(p), dtype=bool)
            elegidos[np.argsort(-p, kind="stable")[:CORTE]] = True
            for seg, mascara in [("cobra el sueldo en el banco", con_sueldo), ("no cobra el sueldo en el banco", ~con_sueldo)]:
                sel = elegidos & mascara
                filas.append({"semilla": s, "metodo": metodo, "segmento": seg, "elegidos": sel.sum(),
                              "baja2": (sel & baja2).sum(), "ganancia": valor[sel].sum()})
    d = pd.DataFrame(filas)
    tabla = []
    for seg, g in d.groupby("segmento", sort=False):
        f0 = g[g.metodo == "ninguna"].set_index("semilla")
        f4 = g[g.metodo == "proporciones"].set_index("semilla")
        dif = f4.ganancia - f0.ganancia
        tabla.append({"segmento": seg, "% de los clientes del mes": f"{100 * (con_sueldo if seg.startswith('cobra') else ~con_sueldo).mean():.0f}%",
                      "elegidos por F0": miles(int(round(f0.elegidos.mean()))), "elegidos por F4": miles(int(round(f4.elegidos.mean()))),
                      "BAJA+2 que agarra F0": decimal(f0.baja2.mean(), 1), "BAJA+2 que agarra F4": decimal(f4.baja2.mean(), 1),
                      "ganancia F4 menos F0": millones(dif.mean()), "F4 gana en": f"{(dif > 0).sum()}/{len(dif)}"})
    return pd.DataFrame(tabla)


def grafico(props: pd.DataFrame, n_variables: int) -> None:
    p = props.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh(p["proporción"], p["% de la ganancia de los cortes"], color=COLOR["proporciones"], height=0.6, zorder=3)
    for y, (v, puesto) in enumerate(zip(p["% de la ganancia de los cortes"], p["puesto entre todas las variables"])):
        ax.annotate(f"{decimal(v, 1)}% · puesto {puesto}", (v, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11, color=TINTA)
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, props["% de la ganancia de los cortes"].max() * 1.3)
    ax.set_xlabel("% de la ganancia de los cortes del modelo (nivel, lags y deltas juntos)")
    ax.set_title("¿Qué proporciones usa el modelo de F4?", pad=50)
    ax.text(0, 1.025, f"Media de los modelos finales de F4 ({MES_TEST}). \"Puesto\": lugar de esa variable entre las "
            f"{n_variables} del modelo,\nordenadas por la ganancia de sus cortes.", transform=ax.transAxes,
            fontsize=11, color=TINTA_2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "importancia_f4.png", dpi=160)
    plt.close(fig)


def main() -> None:
    props, grupos, n_variables = importancia()
    seg = segmentos()
    grupos["%"] = grupos["%"].map(lambda v: f"{decimal(v, 1)}%")
    tabla_props = props.assign(**{"% de la ganancia de los cortes": props["% de la ganancia de los cortes"]
                                  .map(lambda v: f"{decimal(v, 1)}%")})
    partes = ["# Por qué funciona F4 (generado por scripts/12_por_que_f4.py)\n",
              "Seguimiento S8, partes 1 y 2 (README). Sin regla de decisión.\n",
              "## 1. Qué usa el modelo\n",
              "Parte de la ganancia de los cortes de los modelos finales de F4, por grupo de variables (cada "
              "variable junta su nivel, lags y deltas):\n", tabla_md(grupos), "",
              "Por proporción:\n", tabla_md(tabla_props), "",
              f"## 2. Dónde gana (mismo corte para las dos: {miles(CORTE)} clientes, {len(SEMILLAS)} semillas)\n",
              tabla_md(seg), ""]
    (OUTPUT_DIR / "por_que_f4.md").write_text("\n".join(partes), encoding="utf-8")
    grafico(props, n_variables)
    print("\n".join(partes))
    print("-> output/por_que_f4.md y output/importancia_f4.png")


if __name__ == "__main__":
    main()
