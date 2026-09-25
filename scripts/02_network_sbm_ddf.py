#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Fase 2 - Capa 1: niveles de eficiencia (Network SBM-DDF con ventanas)
Guia Metodologica Definitiva (v4) - DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 02_network_sbm_ddf.py

Modelo (Fukuyama & Weber, 2010): sistema de red de dos etapas, medida basada en
holguras con funcion de distancia direccional (SBM-DDF), bajo VRS.

    Etapa I  (captacion):  g_personal_r, n_oficinas  -> total_depositos_r
    Etapa II (colocacion): total_depositos_r -> creditos_vigentes_r,
                           ingresos_finan_r (deseados) + cartera_atrasada_r (malo)

Elementos tecnicos:
  * total_depositos_r es el ENLACE de red (producto de la Etapa I e insumo de la
    Etapa II): variable interna, no entra en la funcion objetivo.
  * Disponibilidad DEBIL del bad output segun Kuosmanen (2005): en la Etapa II se
    separa la intensidad activa z de la de reduccion (abatement) u; los productos
    (buenos y malo) usan solo z, y el insumo intermedio usa z+u. VRS: sum(z+u)=1.
  * Vector direccional g: g_inputs = 0 (insumos cuasi-fijos en el corto plazo),
    expansion en productos deseados y contraccion en el bad output. Las holguras
    de los insumos SIGUEN entrando en la funcion objetivo (medida no radial): el
    modelo no equivale a un output-oriented radial.
  * Orientacion: NO orientado por construccion; la direccion de mejora la fija g.

Medida de ineficiencia del sistema (a maximizar), normalizada al estilo SBM:
    rho = 1/2 [ W1 + W2 ]
    W1  = 1/2 ( s_x1/x1o + s_x2/x2o )                      (Etapa I, insumos)
    W2  = 1/3 ( s_y1/y1o + s_y2/y2o + s_b/bo )             (Etapa II, productos)

Puntajes (transformacion SBM, garantiza (0,1]):
    score_global   = 1 / (1 + rho)
    etapa1_score   = 1 / (1 + W1)
    etapa2_score   = 1 / (1 + W2)

Ventanas moviles w=3 sobre 2002-2025 (analisis de ventanas de Asmild et al.,
2004): cada caja-anio se evalua contra el conjunto de referencia de su ventana
(~34-36 pseudo-unidades). Para el reporte principal se usa la VENTANA CENTRAL de
cada anio (aquella en que el anio es el central), de modo que los puntajes son
directamente comparables con la correccion por bootstrap de la Fase 3.

Salida:
    outputs/tables/scores_originales.xlsx
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pulp

# ------------------------------------------------------------------ #
# Rutas y especificacion
# ------------------------------------------------------------------ #
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Libro1.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUT_PATH = OUT_DIR / "scores_originales.xlsx"

INPUTS = ["g_personal_r", "n_oficinas"]
INTERMEDIATE = "total_depositos_r"
GOOD_OUTPUTS = ["creditos_vigentes_r", "ingresos_finan_r"]
BAD_OUTPUT = "cartera_atrasada_r"

WINDOW = 3           # amplitud de ventana movil
YEAR_MIN, YEAR_MAX = 2002, 2025
EPS = 1e-9


# ------------------------------------------------------------------ #
# Construccion de ventanas
# ------------------------------------------------------------------ #
def build_windows(years):
    """Ventanas moviles consecutivas de amplitud w=WINDOW."""
    ys = sorted(years)
    wins = []
    for start in range(ys[0], ys[-1] - WINDOW + 2):
        block = list(range(start, start + WINDOW))
        if all(y in ys for y in block):
            wins.append(tuple(block))
    return wins


def central_window_for_year(year, windows):
    """
    Ventana central de un anio: aquella en la que el anio ocupa la posicion
    media. Para los anios de borde (2002, 2025) se toma la unica/extrema ventana
    que los contiene mas cercana a la posicion central.
    """
    # Ventana cuyo elemento medio es 'year'
    for w in windows:
        if w[len(w) // 2] == year:
            return w
    # Borde inferior: primera ventana que contiene el anio
    for w in windows:
        if year in w:
            return w
    return None


# ------------------------------------------------------------------ #
# Modelo Network SBM-DDF (PuLP), VRS, disponibilidad debil (Kuosmanen 2005)
# ------------------------------------------------------------------ #
def solve_nsbm_ddf(ref: pd.DataFrame, o: pd.Series):
    """
    Resuelve el modelo de red de dos etapas para la DMU 'o' contra el conjunto de
    referencia 'ref' (pseudo-unidades de la ventana). Devuelve dict con
    score_global, etapa1_score, etapa2_score, rho, W1, W2 y estado del solver.
    """
    n = len(ref)
    idx = list(range(n))

    x1 = ref[INPUTS[0]].to_numpy(float)
    x2 = ref[INPUTS[1]].to_numpy(float)
    dep = ref[INTERMEDIATE].to_numpy(float)
    y1 = ref[GOOD_OUTPUTS[0]].to_numpy(float)
    y2 = ref[GOOD_OUTPUTS[1]].to_numpy(float)
    bb = ref[BAD_OUTPUT].to_numpy(float)

    x1o = float(o[INPUTS[0]]); x2o = float(o[INPUTS[1]])
    y1o = float(o[GOOD_OUTPUTS[0]]); y2o = float(o[GOOD_OUTPUTS[1]])
    bo = float(o[BAD_OUTPUT])

    m = pulp.LpProblem("NSBM_DDF", pulp.LpMaximize)

    # Intensidades
    lam1 = [pulp.LpVariable(f"lam1_{j}", lowBound=0) for j in idx]  # Etapa I
    z2 = [pulp.LpVariable(f"z2_{j}", lowBound=0) for j in idx]      # Etapa II activa
    u2 = [pulp.LpVariable(f"u2_{j}", lowBound=0) for j in idx]      # abatement (Kuosmanen)

    # Holguras
    s_x1 = pulp.LpVariable("s_x1", lowBound=0)
    s_x2 = pulp.LpVariable("s_x2", lowBound=0)
    s_y1 = pulp.LpVariable("s_y1", lowBound=0)
    s_y2 = pulp.LpVariable("s_y2", lowBound=0)
    s_b = pulp.LpVariable("s_b", lowBound=0)

    # Objetivo: rho = 1/2[ 1/2(s_x1/x1o+s_x2/x2o) + 1/3(s_y1/y1o+s_y2/y2o+s_b/bo) ]
    m += (
        0.25 * (s_x1 / x1o + s_x2 / x2o)
        + (1.0 / 6.0) * (s_y1 / y1o + s_y2 / y2o + s_b / bo)
    )

    # --- Etapa I (captacion), VRS ---
    m += pulp.lpSum(lam1[j] * x1[j] for j in idx) + s_x1 == x1o, "I_x1"
    m += pulp.lpSum(lam1[j] * x2[j] for j in idx) + s_x2 == x2o, "I_x2"
    m += pulp.lpSum(lam1[j] for j in idx) == 1, "I_vrs"

    # --- Enlace de red: deposit producido (Etapa I) >= consumido (Etapa II) ---
    m += (
        pulp.lpSum(lam1[j] * dep[j] for j in idx)
        - pulp.lpSum((z2[j] + u2[j]) * dep[j] for j in idx)
        >= 0
    ), "link_dep"

    # --- Etapa II (colocacion), VRS + disponibilidad debil (Kuosmanen 2005) ---
    # Productos deseados: solo intensidad activa z
    m += pulp.lpSum(z2[j] * y1[j] for j in idx) - s_y1 == y1o, "II_y1"
    m += pulp.lpSum(z2[j] * y2[j] for j in idx) - s_y2 == y2o, "II_y2"
    # Bad output: igualdad (weak disposability), solo z
    m += pulp.lpSum(z2[j] * bb[j] for j in idx) + s_b == bo, "II_b"
    # VRS con abatement: sum(z+u) = 1
    m += pulp.lpSum(z2[j] + u2[j] for j in idx) == 1, "II_vrs"

    m.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[m.status]
    if status != "Optimal":
        return {
            "status": status, "score_global": np.nan,
            "etapa1_score": np.nan, "etapa2_score": np.nan,
            "rho": np.nan, "W1": np.nan, "W2": np.nan,
        }

    sx1 = s_x1.value() or 0.0; sx2 = s_x2.value() or 0.0
    sy1 = s_y1.value() or 0.0; sy2 = s_y2.value() or 0.0
    sb = s_b.value() or 0.0

    W1 = 0.5 * (sx1 / x1o + sx2 / x2o)
    W2 = (1.0 / 3.0) * (sy1 / y1o + sy2 / y2o + sb / bo)
    rho = 0.5 * (W1 + W2)

    return {
        "status": status,
        "score_global": 1.0 / (1.0 + rho),
        "etapa1_score": 1.0 / (1.0 + W1),
        "etapa2_score": 1.0 / (1.0 + W2),
        "rho": rho, "W1": W1, "W2": W2,
    }


# ------------------------------------------------------------------ #
# Estimacion de todas las ventanas (analisis de ventanas completo)
# ------------------------------------------------------------------ #
def estimate_all_windows(df, windows):
    """Puntaje de cada caja-anio en CADA ventana en la que aparece."""
    records = []
    for w in windows:
        ref = df[df["anio"].isin(w)].reset_index(drop=True)
        obs = df[df["anio"].isin(w)]
        for _, o in obs.iterrows():
            r = solve_nsbm_ddf(ref, o)
            records.append(
                {
                    "cmac": o["cmac"], "anio": int(o["anio"]),
                    "ventana": f"{w[0]}-{w[-1]}",
                    "ventana_ini": w[0], "ventana_fin": w[-1],
                    "score_global": r["score_global"],
                    "etapa1_score": r["etapa1_score"],
                    "etapa2_score": r["etapa2_score"],
                    "status": r["status"],
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[02] Cargando panel: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name=0).sort_values(["anio", "cmac"])
    df = df.reset_index(drop=True)

    years = sorted(df["anio"].unique())
    windows = build_windows(years)
    print(f"[02] Ventanas w={WINDOW}: {len(windows)}  "
          f"[{windows[0][0]}-{windows[0][-1]} ... {windows[-1][0]}-{windows[-1][-1]}]")

    # 1) Analisis de ventanas completo (todas las apariciones)
    print("[02] Estimando SBM-DDF de red en todas las ventanas (VRS, Kuosmanen)...")
    allw = estimate_all_windows(df, windows)
    n_fail = int((allw["status"] != "Optimal").sum())
    print(f"[02] Evaluaciones: {len(allw)} | no-optimas: {n_fail}")

    # 2) Reporte principal: ventana CENTRAL de cada caja-anio
    cw_map = {y: central_window_for_year(y, windows) for y in years}
    central_rows = []
    for _, o in df.iterrows():
        w = cw_map[int(o["anio"])]
        tag = f"{w[0]}-{w[-1]}"
        row = allw[
            (allw["cmac"] == o["cmac"])
            & (allw["anio"] == int(o["anio"]))
            & (allw["ventana"] == tag)
        ].iloc[0]
        central_rows.append(
            {
                "cmac": o["cmac"], "anio": int(o["anio"]),
                "ventana_central": tag,
                "score_global": row["score_global"],
                "etapa1_score": row["etapa1_score"],
                "etapa2_score": row["etapa2_score"],
            }
        )
    central = pd.DataFrame(central_rows).sort_values(["anio", "cmac"]).reset_index(drop=True)

    # 3) Promedio de ventanas por caja-anio (referencia de analisis de ventanas)
    winavg = (
        allw.groupby(["cmac", "anio"], as_index=False)[
            ["score_global", "etapa1_score", "etapa2_score"]
        ]
        .mean()
        .rename(
            columns={
                "score_global": "score_global_prom_ventanas",
                "etapa1_score": "etapa1_prom_ventanas",
                "etapa2_score": "etapa2_prom_ventanas",
            }
        )
    )

    # 4) Ranking del nucleo (sobre ventana central), promedio por caja
    ranking = (
        central.groupby("cmac", as_index=False)["score_global"]
        .mean()
        .sort_values("score_global", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "puesto", ranking.index + 1)

    # Consola
    print("\n[02] Puntaje global (ventana central) - resumen:")
    print(f"    media={central['score_global'].mean():.4f} | "
          f"min={central['score_global'].min():.4f} | "
          f"max={central['score_global'].max():.4f} | "
          f"n_eficientes(>=0.9999)={(central['score_global']>=0.9999).sum()}")
    print("\n[02] Ranking (promedio caja, ventana central):")
    for _, r in ranking.iterrows():
        print(f"    {int(r['puesto']):2d}. {r['cmac']:<42s} {r['score_global']:.4f}")

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        central.to_excel(xw, sheet_name="scores_ventana_central", index=False)
        allw.to_excel(xw, sheet_name="scores_todas_ventanas", index=False)
        winavg.to_excel(xw, sheet_name="scores_promedio_ventanas", index=False)
        ranking.to_excel(xw, sheet_name="ranking_nucleo", index=False)

    print(f"\n[02] Puntajes originales guardados en: {OUT_PATH}")


if __name__ == "__main__":
    main()
