"""Diagnóstico: ¿cuánto y cómo se mueven las variables en pesos en 2019?

Material para la sección "Hipótesis" del video. Para cada variable en pesos
y cada mes calcula la mediana de |x| entre los clientes con x distinto de 0
(la misma escala que usa la familia F2) y la compara con el IPC, todo como
índice 201903 = 100 (el primer mes de entrenamiento).

Salida:
  output/diagnostico_indices.csv       índice mensual de cada variable y del IPC
  output/diagnostico_medianas_ipc.png  4 variables clave contra el IPC, mes a mes
  output/diagnostico_dispersion.png    variación 201903 -> 201910 de cada variable contra el IPC

Necesita data/base (scripts/01_preparar_base.py).

Uso:
    uv run python scripts/00_diagnostico_inflacion.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from comun import COLUMNAS_CLAVE, es_monetaria, meses_disponibles
from config import IPC, MES_TEST, MES_VALIDACION, MESES_ENTRENAMIENTO, MESES_ENTRENAMIENTO_FINAL, OUTPUT_DIR, ruta_base

BASE = MESES_ENTRENAMIENTO[0]
FONDO, TINTA, TINTA_2, GRILLA = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# slots 5-8 de la paleta de referencia: los 1-4 quedan para las familias
DESTACADAS = {
    "mpayroll": ("Sueldo acreditado", "#e87ba4"),
    "mcuentas_saldo": ("Saldo total en cuentas", "#008300"),
    "mtarjeta_visa_consumo": ("Consumo con Visa", "#6250d6"),
    "Visa_mlimitecompra": ("Límite de compra Visa", "#e34948"),
}

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.8, "legend.frameon": False,
})


def escalas_por_mes() -> pd.DataFrame:
    filas = []
    for mes in meses_disponibles():
        t = pq.read_table(ruta_base(mes)).to_pandas()
        fila = {"foto_mes": mes}
        for c in t.columns:
            if c in COLUMNAS_CLAVE or not es_monetaria(c):
                continue
            x = t[c].to_numpy(dtype=float)
            nz = x[(x != 0) & ~np.isnan(x)]
            fila[c] = np.median(np.abs(nz)) if len(nz) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas).set_index("foto_mes")


def etiqueta_mes(m: int) -> str:
    return ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"][m % 100 - 1]


def separar(valores: dict[str, float], minimo: float) -> dict[str, float]:
    """Posiciones verticales para etiquetas al final de cada línea, separadas
    al menos `minimo` (en unidades del eje) para que no se pisen."""
    orden = sorted(valores, key=valores.get)
    pos = {k: valores[k] for k in orden}
    for _ in range(50):
        for a, b in zip(orden, orden[1:]):
            if pos[b] - pos[a] < minimo:
                centro = (pos[a] + pos[b]) / 2
                pos[a], pos[b] = centro - minimo / 2, centro + minimo / 2
    return pos


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    esc = escalas_por_mes()
    indices = esc / esc.loc[BASE] * 100
    indices.insert(0, "IPC", pd.Series({m: IPC[m] / IPC[BASE] * 100 for m in esc.index}))
    indices.round(2).to_csv(OUTPUT_DIR / "diagnostico_indices.csv")
    rotas = set(pd.read_csv(OUTPUT_DIR / "celdas_rotas.csv").variable) if (OUTPUT_DIR / "celdas_rotas.csv").exists() else set()
    sanas = [c for c in esc.columns if c not in rotas and esc[c].notna().all()]

    # --- 1. cuatro variables clave contra el IPC, mes a mes (desde el primer mes
    #        de entrenamiento: enero y febrero solo se usan para los lags)
    meses = [m for m in indices.index if m >= BASE]
    x = np.arange(len(meses))
    pos = {m: i for i, m in enumerate(meses)}
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axvspan(pos[MESES_ENTRENAMIENTO[0]] - 0.5, pos[MESES_ENTRENAMIENTO[-1]] + 0.5, color="#2a78d6", alpha=0.06, lw=0)
    ax.axvspan(pos[MESES_ENTRENAMIENTO[-1]] + 0.5, pos[MESES_ENTRENAMIENTO_FINAL[-1]] + 0.5, color="#2a78d6",
               alpha=0.03, lw=0)
    ax.axvspan(pos[MES_TEST] - 0.5, pos[MES_TEST] + 0.5, color=TINTA, alpha=0.07, lw=0)
    centro_entrenamiento = (pos[MESES_ENTRENAMIENTO[0]] + pos[MESES_ENTRENAMIENTO[-1]]) / 2
    for texto, xm in (("entrenamiento", centro_entrenamiento), ("+ reentreno (valid. ago)", 4.5),
                      ("test", pos[MES_TEST])):
        ax.text(xm, 1.01, texto, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=11,
                color=TINTA_2)
    finales = {c: indices.loc[meses[-1], c] for c in DESTACADAS} | {"IPC": indices.loc[meses[-1], "IPC"]}
    y_etiqueta = separar(finales, minimo=4.5)
    for c, (nombre, color) in DESTACADAS.items():
        ax.plot(x, indices.loc[meses, c], color=color, lw=2.2, marker="o", ms=5)
        ax.annotate(nombre, (x[-1], indices.loc[meses[-1], c]), xytext=(x[-1] + 0.35, y_etiqueta[c]),
                    textcoords="data", va="center", fontsize=12, color=TINTA)
    ax.plot(x, indices.loc[meses, "IPC"], color=TINTA, lw=2.6, ls="--")
    ax.annotate("IPC (INDEC)", (x[-1], indices.loc[meses[-1], "IPC"]), xytext=(x[-1] + 0.35, y_etiqueta["IPC"]),
                textcoords="data", va="center", fontsize=12, color=TINTA, fontweight="bold")
    ax.set_xticks(x, [etiqueta_mes(m) for m in meses])
    ax.set_xlim(-0.5, len(meses) + 1.6)
    ax.set_ylabel(f"Índice ({etiqueta_mes(BASE)} 2019 = 100)")
    ax.set_title("En 2019 cada variable en pesos tiene su propia inflación", pad=56)
    ax.text(0, 1.085, "Mediana mensual entre los clientes que tienen el producto, contra el IPC. "
            "Junio: aguinaldo. Agosto: devaluación post-PASO.", transform=ax.transAxes, fontsize=11, color=TINTA_2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostico_medianas_ipc.png", dpi=160)
    plt.close(fig)

    # --- 2. variación entre el inicio del entrenamiento y el test, variable por variable
    var = (indices.loc[MES_TEST, sanas] / 100 - 1) * 100
    var = var.sort_values()
    ipc = (IPC[MES_TEST] / IPC[BASE] - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 5.0))
    rng = np.random.default_rng(0)
    y = rng.uniform(-0.25, 0.25, len(var))
    ax.scatter(var, y, s=55, color="#6d7378", alpha=0.8, edgecolor=FONDO, lw=1, zorder=3)
    ax.axvline(ipc, color=TINTA, lw=2.4, ls="--")
    ax.annotate(f" IPC +{ipc:.0f}%", (ipc, 0.45), ha="left", fontsize=12, fontweight="bold")
    ax.axvline(var.median(), color="#1baf7a", lw=2.4)
    ax.annotate(f"mediana de las variables +{var.median():.0f}% ", (var.median(), -0.5), ha="right", fontsize=12,
                color=TINTA)
    for c in [var.index[0], var.index[-1], "mcuentas_saldo", "mpayroll"]:
        if c in var.index:
            ax.annotate(c, (var[c], y[list(var.index).index(c)]), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=10, color=TINTA_2)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Variación de la mediana mensual entre {etiqueta_mes(BASE)} y {etiqueta_mes(MES_TEST)} 2019 (%)")
    ax.set_title(f"Un solo índice no alcanza: {len(var)} variables en pesos, {len(var)} inflaciones distintas", pad=28)
    ax.text(0, 1.03, "Cada punto es una variable en pesos (sin las que tienen meses rotos). "
            f"La mitad de las variables se movió entre {var.quantile(0.25):+.0f}% y {var.quantile(0.75):+.0f}%.",
            transform=ax.transAxes, fontsize=11, color=TINTA_2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diagnostico_dispersion.png", dpi=160)
    plt.close(fig)

    print(f"IPC {etiqueta_mes(BASE)} -> {etiqueta_mes(MES_TEST)}: +{ipc:.1f}%")
    print(f"Variables en pesos sanas: {len(var)}; variación mediana +{var.median():.1f}%, "
          f"p10 {var.quantile(0.1):+.1f}%, p90 {var.quantile(0.9):+.1f}%")
    for c in DESTACADAS:
        print(f"  {c}: " + ", ".join(f"{etiqueta_mes(m)} {indices.loc[m, c]:.0f}" for m in meses))
    print(f"-> {OUTPUT_DIR / 'diagnostico_medianas_ipc.png'}, {OUTPUT_DIR / 'diagnostico_dispersion.png'}")


if __name__ == "__main__":
    main()
