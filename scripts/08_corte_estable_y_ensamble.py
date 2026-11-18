"""Corte estable y ensamble del nominal con una corrección (seguimiento S1 y S2).

Reglas (escritas en el README antes de correr):
  S1. El corte de cada familia es el máximo de la curva de ganancia PROMEDIO
      de sus 20 semillas en 201908 (modelos de 07, entrenados con
      201903-201906), en vez del corte de un solo modelo (el mejor intento de
      la bayesiana). Se aplica a las 20 semillas finales en 201910 (de 04).
      Contra F0, también con su corte estable: una familia supera a F0 si la
      diferencia media es mayor que el desvío de F0 y el Wilcoxon pareado por
      semilla da p < 0,05.
  S2. Para cada semilla se promedian las probabilidades de F0 y de otra
      familia. Control: F0 con esa semilla + F0 con la semilla siguiente. El
      corte de cada ensamble es su corte estable en 201908 (como en S1). Un
      ensamble supera al control si la diferencia media en 201910 es mayor que
      el desvío del control y el Wilcoxon pareado da p < 0,05.

Necesita las predicciones locales de 04 y 07 (resultados/prediccion*.parquet).

Salida:
  output/corte_estable_y_ensamble.md
  output/cortes_validacion.png       corte de cada semilla contra el corte estable
  output/ensamble_vs_control.png     diferencia de cada ensamble contra el control
  resultados/corte_estable_y_ensamble.csv   ganancia en 201910 por semilla

Uso:
    uv run python scripts/08_corte_estable_y_ensamble.py
"""
from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from comun import cortes_barrido, ganancia_por_corte, leer_bo
from config import MES_TEST, MES_VALIDACION, METODOS, NOMBRE_FAMILIA, OUTPUT_DIR, RESULTADOS_DIR, SEMILLAS, ruta_base

r6 = importlib.import_module("06_resumen")  # mismo formato, colores y comparación pareada que el resumen
millones, miles, decimal, formato_p, tabla_md, comparar = (r6.millones, r6.miles, r6.decimal, r6.formato_p,
                                                           r6.tabla_md, r6.comparar)
COLOR, FONDO, TINTA, TINTA_2 = r6.COLOR, r6.FONDO, r6.TINTA, r6.TINTA_2
CORTES = cortes_barrido()
CONTROL = "F0 + F0 (otra semilla)"


def curvas(prefijo: str, metodo: str) -> pd.DataFrame:
    """Ganancia por corte de cada semilla (filas = semillas, columnas = cortes)."""
    filas = {s: pd.read_csv(p).set_index("corte")["ganancia"]
             for s in SEMILLAS if (p := RESULTADOS_DIR / f"{prefijo}__{metodo}__{s}.csv").exists()}
    return pd.DataFrame(filas).T


def corte_estable(V: pd.DataFrame) -> int:
    return int(V.mean().idxmax())


def predicciones(prefijo: str, metodo: str, semillas: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    clientes, probs = None, {}
    for s in semillas:
        p = pd.read_parquet(RESULTADOS_DIR / f"{prefijo}__{metodo}__{s}.parquet")
        if clientes is None:
            clientes = p.numero_de_cliente.to_numpy()
        elif not np.array_equal(clientes, p.numero_de_cliente.to_numpy()):
            raise SystemExit(f"{prefijo}__{metodo}__{s}: los clientes no están en el mismo orden")
        probs[s] = p.prob.to_numpy()
    return clientes, probs


def clase_de(mes: int, clientes: np.ndarray) -> np.ndarray:
    base = pq.read_table(ruta_base(mes), columns=["numero_de_cliente", "clase_ternaria"]).to_pandas()
    return base.set_index("numero_de_cliente").loc[clientes, "clase_ternaria"].to_numpy()


def tiene_predicciones(metodo: str, s: int) -> bool:
    return all((RESULTADOS_DIR / f"{p}__{metodo}__{s}.parquet").exists()
               for p in ("prediccion", "prediccion_validacion"))


# ---------------------------------------------------------------------------
# S1: corte estable por familia
# ---------------------------------------------------------------------------
def s1() -> tuple[pd.DataFrame, dict]:
    filas, info = [], {}
    for m in METODOS:
        V, T = curvas("curva_validacion", m), curvas("curva", m)
        if V.empty or T.empty:
            continue
        comunes = V.index.intersection(T.index)
        V, T = V.loc[comunes], T.loc[comunes]
        c, c_bo = corte_estable(V), int(leer_bo(m)["corte_validacion"])
        info[m] = {"corte_bo": c_bo, "corte_estable": c, "cortes_semilla": V.idxmax(axis=1).astype(int),
                   "ganancia_bo": T[c_bo], "validacion_en_corte": V[c]}
        filas += [{"grupo": "S1", "metodo": m, "semilla": s, "corte": c, "ganancia": int(T.loc[s, c])}
                  for s in comunes]
    return pd.DataFrame(filas), info


# ---------------------------------------------------------------------------
# S2: ensamble F0 + otra familia, contra F0 + F0 con otra semilla
# ---------------------------------------------------------------------------
def s2() -> tuple[pd.DataFrame, dict]:
    familias = [m for m in METODOS if any(tiene_predicciones(m, s) for s in SEMILLAS)]
    if "ninguna" not in familias:
        return pd.DataFrame(), {}
    semillas_f0 = [s for s in SEMILLAS if tiene_predicciones("ninguna", s)]
    pv, pt, cv, ct = {}, {}, None, None
    for m in familias:
        sems = [s for s in SEMILLAS if tiene_predicciones(m, s)]
        cv_m, pv[m] = predicciones("prediccion_validacion", m, sems)
        ct_m, pt[m] = predicciones("prediccion", m, sems)
        if cv is None:
            cv, ct = cv_m, ct_m
        elif not (np.array_equal(cv, cv_m) and np.array_equal(ct, ct_m)):
            raise SystemExit(f"{m}: los clientes no coinciden con los de F0")
    clase_v, clase_t = clase_de(MES_VALIDACION, cv), clase_de(MES_TEST, ct)

    siguiente = {s: semillas_f0[(i + 1) % len(semillas_f0)] for i, s in enumerate(semillas_f0)}
    socios = {CONTROL: ("ninguna", lambda s: siguiente[s])}
    socios |= {f"F0 + {NOMBRE_FAMILIA[m]}": (m, lambda s: s) for m in familias if m != "ninguna"}

    filas, info = [], {}
    for nombre, (m, semilla_socio) in socios.items():
        sems = [s for s in semillas_f0 if semilla_socio(s) in pv[m]]
        if not sems:
            continue
        V = pd.DataFrame([ganancia_por_corte((pv["ninguna"][s] + pv[m][semilla_socio(s)]) / 2, clase_v, CORTES)
                          for s in sems], index=sems, columns=CORTES)
        T = pd.DataFrame([ganancia_por_corte((pt["ninguna"][s] + pt[m][semilla_socio(s)]) / 2, clase_t, CORTES)
                          for s in sems], index=sems, columns=CORTES)
        c = corte_estable(V)
        info[nombre] = {"metodo": m, "corte_estable": c, "validacion_en_corte": V[c]}
        filas += [{"grupo": "S2", "metodo": nombre, "semilla": s, "corte": c, "ganancia": int(T.loc[s, c])}
                  for s in sems]
    return pd.DataFrame(filas), info


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_cortes(info: dict) -> None:
    familias = list(info)
    fig, ax = plt.subplots(figsize=(11, 1.9 + 1.0 * len(familias)))
    for i, m in enumerate(familias):
        cs = info[m]["cortes_semilla"].to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(cs))
        ax.scatter(cs, i + jitter, s=45, color=COLOR[m], alpha=0.85, edgecolor=FONDO, lw=1, zorder=3)
        ax.plot([info[m]["corte_bo"]], [i], "o", ms=13, mfc="none", mec=TINTA, mew=2, zorder=4)
        ax.plot([info[m]["corte_estable"]] * 2, [i - 0.32, i + 0.32], color=TINTA, lw=2.8, zorder=5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: miles(int(v))))
    ax.set_yticks(range(len(familias)), [NOMBRE_FAMILIA[m] for m in familias])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Clientes a estimular que maximizan la ganancia en {MES_VALIDACION}")
    ax.set_title(f"Corte elegido en {MES_VALIDACION}: un modelo por semilla contra el promedio", pad=50)
    ax.text(0, 1.025, "Punto = corte de cada semilla. Anillo = el de la bayesiana (un solo modelo, el que se usó).\n"
            "Raya = corte estable: el máximo de la curva promedio de las semillas.", transform=ax.transAxes,
            fontsize=11, color=TINTA_2, va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cortes_validacion.png", dpi=160)
    plt.close(fig)


def grafico_ensambles(tabla: pd.DataFrame, info: dict) -> None:
    control = tabla[tabla.metodo == CONTROL].set_index("semilla")["ganancia"]
    otros = [n for n in info if n != CONTROL]
    fig, ax = plt.subplots(figsize=(11, 1.6 + 1.3 * len(otros)))
    for i, nombre in enumerate(otros):
        g = tabla[tabla.metodo == nombre].set_index("semilla")["ganancia"]
        comunes = g.index.intersection(control.index)
        dif = (g.loc[comunes] - control.loc[comunes]) / 1e6
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(dif))
        ax.scatter(dif, i + jitter, s=60, color=COLOR[info[nombre]["metodo"]], edgecolor=FONDO, linewidth=1.5,
                   zorder=3)
        ax.plot([dif.mean()] * 2, [i - 0.3, i + 0.3], color=TINTA, lw=2.5, zorder=4)
        r = comparar(tabla, nombre, CONTROL, "ganancia")
        p = r["p_wilcoxon"]
        p_txt = "" if np.isnan(p) else " · p < 0,001" if p < 0.001 else f" · p = {decimal(p, 3)}"
        ax.annotate(f"media {millones(r['dif_media'])} · gana en {r['gana_en']}/{r['semillas']}{p_txt}",
                    (1.0, i), xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=11, color=TINTA_2)
    umbral = control.std() / 1e6
    izq, der = ax.get_xlim()
    der = max(der, umbral + 1)
    ax.axvspan(umbral, der, color=TINTA, alpha=0.06, lw=0, zorder=0)
    ax.set_xlim(izq, der)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
    ax.axvline(0, color=TINTA_2, lw=1.2, ls="--")
    ax.set_yticks(range(len(otros)), otros)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Ganancia del ensamble menos la del control, con la misma semilla (millones de $)")
    ax.set_title(f"Ensamble de F0 con otra familia contra F0 + F0 ({MES_TEST})", pad=50)
    ax.text(0, 1.025, "Cada punto es una semilla; la raya negra es la media. Cada ensamble usa su corte estable.\n"
            f"Zona gris: para superar al control, la media tiene que pasar 1 desvío del control "
            f"({millones(umbral * 1e6)}).", transform=ax.transAxes, fontsize=11, color=TINTA_2, va="bottom")
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)
    fig.savefig(OUTPUT_DIR / "ensamble_vs_control.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    t1, info1 = s1()
    if t1.empty:
        raise SystemExit("Faltan resultados: correr antes scripts/04_entrenar_final.py y 07_validacion_semillas.py")
    t2, info2 = s2()
    pd.concat([t1, t2], ignore_index=True).to_csv(RESULTADOS_DIR / "corte_estable_y_ensamble.csv", index=False)

    partes = ["# Corte estable y ensamble (generado por scripts/08_corte_estable_y_ensamble.py)\n",
              "Experimentos de seguimiento S1 y S2 (reglas en el README). Ganancia en 201910; ± es el desvío entre "
              "semillas.\n"]
    filas = []
    for m, d in info1.items():
        g = t1[t1.metodo == m].ganancia
        fila = {"familia": NOMBRE_FAMILIA[m], "semillas": len(g),
                "corte de la bayesiana": miles(d["corte_bo"]), "corte estable": miles(d["corte_estable"]),
                "cortes por semilla (mín. a máx.)": f"{miles(int(d['cortes_semilla'].min()))} a "
                                                    f"{miles(int(d['cortes_semilla'].max()))}",
                "ganancia con el corte de la bayesiana": millones(d["ganancia_bo"].mean()),
                "ganancia con el corte estable": f"{millones(g.mean())} ± {millones(g.std())}"}
        if m != "ninguna" and "ninguna" in info1:
            r = comparar(t1, m, "ninguna", "ganancia")
            supera = r["dif_media"] > r["desvio_contra"] and r["p_wilcoxon"] < 0.05
            fila |= {"dif. vs F0": millones(r["dif_media"]), "gana en": f"{r['gana_en']}/{r['semillas']}",
                     "p Wilcoxon": formato_p(r["p_wilcoxon"]), "¿supera a F0?": "sí" if supera else "no"}
        filas.append(fila)
    partes += [f"## S1. Corte estable (máximo de la curva promedio de las semillas en {MES_VALIDACION})\n",
               tabla_md(pd.DataFrame(filas).fillna("")), ""]
    grafico_cortes(info1)

    if info2:
        f0_solo = t1[t1.metodo == "ninguna"].set_index("semilla").ganancia if "ninguna" in info1 else None
        filas = []
        for nombre, d in info2.items():
            g = t2[t2.metodo == nombre].set_index("semilla").ganancia
            fila = {"ensamble": nombre, "semillas": len(g), "corte estable": miles(d["corte_estable"]),
                    "ganancia 201908 en su corte": millones(d["validacion_en_corte"].mean()),
                    "ganancia 201910": f"{millones(g.mean())} ± {millones(g.std())}"}
            if nombre != CONTROL:
                r = comparar(t2, nombre, CONTROL, "ganancia")
                supera = r["dif_media"] > r["desvio_contra"] and r["p_wilcoxon"] < 0.05
                fila |= {"dif. vs control": millones(r["dif_media"]), "gana en": f"{r['gana_en']}/{r['semillas']}",
                         "p Wilcoxon": formato_p(r["p_wilcoxon"]), "¿supera al control?": "sí" if supera else "no"}
            if f0_solo is not None:
                comunes = g.index.intersection(f0_solo.index)
                fila["dif. vs F0 solo"] = millones((g.loc[comunes] - f0_solo.loc[comunes]).mean())
            filas.append(fila)
        orden = ["ensamble", "semillas", "corte estable", "ganancia 201908 en su corte", "ganancia 201910",
                 "dif. vs control", "gana en", "p Wilcoxon", "¿supera al control?", "dif. vs F0 solo"]
        tabla2 = pd.DataFrame(filas)
        tabla2 = tabla2[[c for c in orden if c in tabla2.columns]].fillna("")
        partes += ["## S2. Ensamble F0 + otra familia (promedio de probabilidades, misma semilla)\n",
                   "Control: F0 con la misma semilla + F0 con la semilla siguiente. \"dif. vs F0 solo\": contra F0 "
                   "con su corte estable (S1).\n", tabla_md(tabla2), ""]
        if len(info2) > 1:
            grafico_ensambles(t2, info2)

    (OUTPUT_DIR / "corte_estable_y_ensamble.md").write_text("\n".join(partes), encoding="utf-8")
    print("\n".join(partes))


if __name__ == "__main__":
    main()
