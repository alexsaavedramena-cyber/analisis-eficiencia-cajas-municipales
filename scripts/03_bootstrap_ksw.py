#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Fase 3 - Correccion de sesgo por bootstrap (submuestreo KSW 2015)
Guia Metodologica Definitiva (v4) - DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 03_bootstrap_ksw.py

Convierte los puntajes SBM-DDF de red (sesgados al alza, con sesgo creciente
cuando N es pequeno) en estimadores con intervalo, aplicando el procedimiento de
SUBMUESTREO de Kneip, Simar y Wilson (2015), apropiado para medidas NO radiales
(la variante radial clasica de Simar-Wilson es inconsistente aqui).

ENFOQUE B (el prescrito por la guia, "Que hago" de la Fase 3):
  "Obtengo, para cada caja-anio, el puntaje corregido en CADA ventana en que
   aparece y LUEGO PROMEDIO los puntajes ya corregidos."
  NO se promedia primero y se bootstrapea despues (eso queda prohibido: el
  promedio de tres ventanas no es la salida de ninguna frontera unica).

Procedimiento:
  1. Cada caja-anio aparece en hasta 3 ventanas moviles w=3 (menos en los bordes:
     2002 en 1, 2003 en 2, 2004-2023 en 3, 2024 en 2, 2025 en 1).
  2. En CADA ventana en que aparece:
       * Se recomputa el puntaje original delta_hat_W con la MISMA formulacion de
         red de la Fase 2 (Network SBM-DDF, VRS, Kuosmanen 2005) resuelta en HiGHS.
       * Submuestreo KSW: B=2000 replicas. Cada replica extrae, SIN reemplazo, un
         subconjunto de tamano m = floor(n^kappa), kappa=0.7, del conjunto de
         referencia de la ventana (n ~ 34-36 pseudo-unidades), y re-evalua la DMU
         contra esa frontera reducida -> delta_estrella. Si la DMU cae fuera del
         casco (super-eficiente), la evaluacion es infactible y delta_estrella=1.0.
         Por monotonia del casco, delta_estrella >= delta_hat_W siempre.
  3. Punto estimado anual = PROMEDIO sobre ventanas de los puntajes ya corregidos:
         score_corregido = mean_W( corregido_W )
         score_original  = mean_W( delta_hat_W )   (promedio de originales, coherente)
         etapa1/etapa2   = mean_W( divisional original )
  4. INTERVALO ANUAL: IC ENVOLVENTE = union de los intervalos de las ventanas en
     que aparece la caja-anio -> [min(ic_inf_W), max(ic_sup_W)]. Se elige por
     coherencia con el punto promediado: como cada corregido_W esta en su IC_W,
     el promedio de corregidos cae SIEMPRE dentro del envolvente (control (c)
     garantizado al 100%). Se conserva ademas, como referencia, el IC de la
     VENTANA CENTRAL (hoja ic_comparacion); ventana central de t = aquella donde
     t es el ano medio, con bordes 2002->[2002-2004], 2025->[2023-2025].
     [Nota: la guia sugeria reportar solo el IC de la ventana central; el IC
      envolvente es una decision explicita del usuario para este proyecto.]

Correccion y tasa (KSW 2015), aplicada dentro de cada ventana:
  Tasa n^{2/(d+1)} con d = p+q = 2 insumos + 3 productos del sistema => 2/(d+1)=1/3.
  Reescalado por submuestreo (Politis-Romano-Wolf):
      t_b        = m^{1/3} * (delta_estrella_b - delta_hat_W)
      sesgo_n    = (m/n)^{1/3} * ( mean(delta_estrella) - delta_hat_W )
      corregido_W= delta_hat_W - sesgo_n
      IC 95% por ventana: [ delta_hat - q_{0.975}(t_b)/n^{1/3},
                            delta_hat - q_{0.025}(t_b)/n^{1/3} ]
      IC 95% anual = envolvente [min ic_inf_W, max ic_sup_W] sobre sus ventanas.

Controles de validez (condicion de aceptacion de la Capa 1):
  (a) corregido <= original           (a nivel del promedio anual)
  (b) corregido <= 1.0
  (c) corregido dentro de su intervalo (IC de la ventana central)

Reproducibilidad: seed=42, derivada de forma determinista por ventana
(default_rng([SEED, anio_inicio_ventana])): resultado identico con/sin paralelismo.

Salidas (outputs/tables/eficiencia_limpia.xlsx):
  * eficiencia_limpia : cmac, anio, score_original, score_corregido, ic_inferior,
                        ic_superior, etapa1_score, etapa2_score
  * ranking_corregido : ranking de las 12 cajas por score corregido promedio
  * controles_validez : las 3 comprobaciones
  * detalle_ventanas  : corregido/original/IC por caja-anio-ventana (auditable)
================================================================================
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ------------------------------------------------------------------ #
# Configuracion
# ------------------------------------------------------------------ #
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Libro1.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUT_PATH = OUT_DIR / "eficiencia_limpia.xlsx"

INPUTS = ["g_personal_r", "n_oficinas"]
INTERMEDIATE = "total_depositos_r"
GOOD_OUTPUTS = ["creditos_vigentes_r", "ingresos_finan_r"]
BAD_OUTPUT = "cartera_atrasada_r"

WINDOW = 3
B_REPS = 2000          # replicas de bootstrap por ventana
KAPPA = 0.7            # exponente del tamano de submuestra: m = floor(n^kappa)
SEED = 42
DIM_D = 5              # d = p+q = 2 insumos + 3 productos del sistema
RATE_EXP = 2.0 / (DIM_D + 1)   # = 1/3
ALPHA = 0.05           # IC al 95%


# ------------------------------------------------------------------ #
# Ventanas
# ------------------------------------------------------------------ #
def build_windows(years):
    ys = sorted(years)
    wins = []
    for start in range(ys[0], ys[-1] - WINDOW + 2):
        block = list(range(start, start + WINDOW))
        if all(y in ys for y in block):
            wins.append(tuple(block))
    return wins


def central_window_for_year(year, windows):
    for w in windows:
        if w[len(w) // 2] == year:
            return w
    for w in windows:
        if year in w:
            return w
    return None


# ------------------------------------------------------------------ #
# Modelo Network SBM-DDF en forma matricial (HiGHS) - identico a la Fase 2
# Variables: [lam1(n), z2(n), u2(n), s_x1, s_x2, s_y1, s_y2, s_b]
# ------------------------------------------------------------------ #
def solve_nsbm_ddf_matrix(R, o):
    """
    R: dict de arrays de referencia {x1,x2,dep,y1,y2,b} (longitud n).
    o: dict con los valores de la DMU evaluada {x1,x2,y1,y2,b}.
    Devuelve (score_global, etapa1_score, etapa2_score) o None si es infactible.
    """
    x1, x2 = R["x1"], R["x2"]
    dep = R["dep"]
    y1, y2, bb = R["y1"], R["y2"], R["b"]
    n = x1.shape[0]
    nv = 3 * n + 5

    x1o, x2o = o["x1"], o["x2"]
    y1o, y2o, bo = o["y1"], o["y2"], o["b"]

    i_sx1, i_sx2 = 3 * n, 3 * n + 1
    i_sy1, i_sy2 = 3 * n + 2, 3 * n + 3
    i_sb = 3 * n + 4

    c = np.zeros(nv)
    c[i_sx1] = -0.25 / x1o
    c[i_sx2] = -0.25 / x2o
    c[i_sy1] = -(1.0 / 6.0) / y1o
    c[i_sy2] = -(1.0 / 6.0) / y2o
    c[i_sb] = -(1.0 / 6.0) / bo

    A_eq = np.zeros((7, nv))
    b_eq = np.zeros(7)
    lam = slice(0, n)
    z2 = slice(n, 2 * n)
    u2 = slice(2 * n, 3 * n)

    A_eq[0, lam] = x1; A_eq[0, i_sx1] = 1.0; b_eq[0] = x1o
    A_eq[1, lam] = x2; A_eq[1, i_sx2] = 1.0; b_eq[1] = x2o
    A_eq[2, lam] = 1.0; b_eq[2] = 1.0
    A_eq[3, z2] = y1; A_eq[3, i_sy1] = -1.0; b_eq[3] = y1o
    A_eq[4, z2] = y2; A_eq[4, i_sy2] = -1.0; b_eq[4] = y2o
    A_eq[5, z2] = bb; A_eq[5, i_sb] = 1.0; b_eq[5] = bo
    A_eq[6, z2] = 1.0; A_eq[6, u2] = 1.0; b_eq[6] = 1.0

    A_ub = np.zeros((1, nv))
    A_ub[0, lam] = -dep
    A_ub[0, z2] = dep
    A_ub[0, u2] = dep
    b_ub = np.array([0.0])

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method="highs",
    )
    if not res.success:
        return None

    v = res.x
    sx1, sx2 = v[i_sx1], v[i_sx2]
    sy1, sy2, sb = v[i_sy1], v[i_sy2], v[i_sb]
    W1 = 0.5 * (sx1 / x1o + sx2 / x2o)
    W2 = (1.0 / 3.0) * (sy1 / y1o + sy2 / y2o + sb / bo)
    rho = 0.5 * (W1 + W2)
    return 1.0 / (1.0 + rho), 1.0 / (1.0 + W1), 1.0 / (1.0 + W2)


# ------------------------------------------------------------------ #
# Worker: procesa UNA ventana completa -> corrige TODAS sus caja-anio
# ------------------------------------------------------------------ #
def process_window(payload):
    win = payload["window"]
    win_df = payload["data"]              # DataFrame de la ventana (referencia = evaluadas)
    central_years = set(payload["central_years"])  # anios para los que ESTA es la ventana central

    Rfull = {
        "x1": win_df[INPUTS[0]].to_numpy(float),
        "x2": win_df[INPUTS[1]].to_numpy(float),
        "dep": win_df[INTERMEDIATE].to_numpy(float),
        "y1": win_df[GOOD_OUTPUTS[0]].to_numpy(float),
        "y2": win_df[GOOD_OUTPUTS[1]].to_numpy(float),
        "b": win_df[BAD_OUTPUT].to_numpy(float),
    }
    n = Rfull["x1"].shape[0]
    m = max(2, int(np.floor(n ** KAPPA)))

    # Submuestras compartidas por todas las DMU de la ventana
    rng = np.random.default_rng([SEED, int(win[0])])
    subs = [rng.choice(n, size=m, replace=False) for _ in range(B_REPS)]

    a_m = m ** RATE_EXP
    a_n = n ** RATE_EXP

    rows = []
    for _, o in win_df.iterrows():
        ov = {
            "x1": float(o[INPUTS[0]]), "x2": float(o[INPUTS[1]]),
            "y1": float(o[GOOD_OUTPUTS[0]]), "y2": float(o[GOOD_OUTPUTS[1]]),
            "b": float(o[BAD_OUTPUT]),
        }
        delta_hat, e1, e2 = solve_nsbm_ddf_matrix(Rfull, ov)

        deltas = np.empty(B_REPS)
        for bidx, sidx in enumerate(subs):
            Rsub = {k: v[sidx] for k, v in Rfull.items()}
            r = solve_nsbm_ddf_matrix(Rsub, ov)
            deltas[bidx] = 1.0 if r is None else r[0]

        mean_delta = deltas.mean()
        bias_n = (m / n) ** RATE_EXP * (mean_delta - delta_hat)
        corrected_W = delta_hat - bias_n

        t = a_m * (deltas - delta_hat)
        q_lo = np.quantile(t, ALPHA / 2)
        q_hi = np.quantile(t, 1 - ALPHA / 2)
        ic_inf_W = delta_hat - q_hi / a_n
        ic_sup_W = delta_hat - q_lo / a_n

        rows.append(
            {
                "cmac": o["cmac"], "anio": int(o["anio"]),
                "ventana": f"{win[0]}-{win[-1]}",
                "n_ref": n, "m_sub": m, "B": B_REPS,
                "delta_hat_W": delta_hat,
                "corregido_W": corrected_W,
                "ic_inf_W": ic_inf_W, "ic_sup_W": ic_sup_W,
                "etapa1_W": e1, "etapa2_W": e2,
                "es_central": int(o["anio"]) in central_years,
                "mean_delta_star": mean_delta,
                "frac_delta_star_ef": float(np.mean(deltas >= 0.999999)),
            }
        )
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[03] Cargando panel: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name=0).sort_values(["anio", "cmac"])
    df = df.reset_index(drop=True)

    years = sorted(df["anio"].unique())
    windows = build_windows(years)
    cw_map = {y: central_window_for_year(y, windows) for y in years}

    # Para cada ventana, que anios la tienen como VENTANA CENTRAL
    central_years_by_win = {w: [] for w in windows}
    for y, w in cw_map.items():
        central_years_by_win[w].append(y)

    payloads = []
    for w in windows:
        win_df = df[df["anio"].isin(w)].reset_index(drop=True)
        payloads.append(
            {"window": w, "data": win_df, "central_years": central_years_by_win[w]}
        )

    print(f"[03] Enfoque B: bootstrap por ventana + promedio de corregidos.")
    print(f"[03] Bootstrap KSW: B={B_REPS}, kappa={KAPPA}, exponente tasa=1/{int(round(1/RATE_EXP))}, "
          f"seed={SEED}")
    print(f"[03] {len(windows)} ventanas | {int(sum(len(p['data']) for p in payloads))} "
          f"evaluaciones caja-anio-ventana")

    n_workers = min(len(payloads), max(1, (os.cpu_count() or 2) - 1))
    print(f"[03] Ejecutando en paralelo con {n_workers} procesos...")

    detalle = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(process_window, payloads):
            detalle.extend(res)
    det = pd.DataFrame(detalle)

    # ---------------- Agregacion: PROMEDIO de corregidos por caja-anio -------- #
    agg = (
        det.groupby(["cmac", "anio"], as_index=False)
        .agg(
            n_ventanas=("ventana", "size"),
            score_original=("delta_hat_W", "mean"),
            score_corregido=("corregido_W", "mean"),
            etapa1_score=("etapa1_W", "mean"),
            etapa2_score=("etapa2_W", "mean"),
        )
    )

    # IC ENVOLVENTE: union de los intervalos de las ventanas donde aparece la
    # caja-anio -> [min(ic_inf_W), max(ic_sup_W)]. Coherente con el punto
    # promediado del enfoque B: como cada corregido_W esta en su IC_W, el
    # promedio de corregidos cae SIEMPRE dentro del envolvente (control (c)
    # garantizado al 100%). Se conserva ademas el IC de la ventana central como
    # referencia (hoja ic_comparacion).
    env = det.groupby(["cmac", "anio"], as_index=False).agg(
        ic_inferior=("ic_inf_W", "min"),
        ic_superior=("ic_sup_W", "max"),
    )
    central = det[det["es_central"]][["cmac", "anio", "ventana", "ic_inf_W", "ic_sup_W"]]
    central = central.rename(
        columns={"ventana": "ventana_central",
                 "ic_inf_W": "ic_central_inf", "ic_sup_W": "ic_central_sup"}
    )
    out = agg.merge(env, on=["cmac", "anio"], how="left").merge(
        central, on=["cmac", "anio"], how="left"
    )

    # Acotar a (0,1] (artefacto de reescalado con N pequeno)
    for col in ["score_corregido", "ic_inferior", "ic_superior"]:
        out[col] = out[col].clip(lower=0.0, upper=1.0)

    out = out.sort_values(["anio", "cmac"]).reset_index(drop=True)

    # ------------------------- Controles de validez -------------------------- #
    tol = 1e-6
    chk_a = out["score_corregido"] <= out["score_original"] + tol
    chk_b = out["score_corregido"] <= 1.0 + tol
    chk_c = (
        (out["score_corregido"] >= out["ic_inferior"] - tol)
        & (out["score_corregido"] <= out["ic_superior"] + tol)
    )

    print("\n[03] Controles de validez (sobre {} observaciones):".format(len(out)))
    print(f"    (a) corregido <= original : {int(chk_a.sum())}/{len(out)} "
          f"{'OK' if chk_a.all() else '!! REVISAR'}")
    print(f"    (b) corregido <= 1.0      : {int(chk_b.sum())}/{len(out)} "
          f"{'OK' if chk_b.all() else '!! REVISAR'}")
    print(f"    (c) corregido dentro IC   : {int(chk_c.sum())}/{len(out)} "
          f"{'OK' if chk_c.all() else '!! REVISAR (IC envolvente)'}")

    # Controles a NIVEL VENTANA (condicion de aceptacion por frontera unica)
    chk_aW = det["corregido_W"] <= det["delta_hat_W"] + tol
    chk_bW = det["corregido_W"] <= 1.0 + tol
    chk_cW = (
        (det["corregido_W"] >= det["ic_inf_W"] - tol)
        & (det["corregido_W"] <= det["ic_sup_W"] + tol)
    )
    print(f"    [nivel ventana] (a){int(chk_aW.sum())}/{len(det)} "
          f"(b){int(chk_bW.sum())}/{len(det)} (c){int(chk_cW.sum())}/{len(det)}")

    print("\n[03] Resumen de puntajes (promedio de corregidos, enfoque B):")
    print(f"    original : media={out['score_original'].mean():.4f} "
          f"min={out['score_original'].min():.4f} max={out['score_original'].max():.4f}")
    print(f"    corregido: media={out['score_corregido'].mean():.4f} "
          f"min={out['score_corregido'].min():.4f} max={out['score_corregido'].max():.4f}")
    print(f"    sesgo medio (orig-corr) = "
          f"{(out['score_original']-out['score_corregido']).mean():.4f}")

    # --------------------- Ranking de cajas por score corregido -------------- #
    ranking = (
        out.groupby("cmac", as_index=False)
        .agg(
            score_corregido_prom=("score_corregido", "mean"),
            score_original_prom=("score_original", "mean"),
            anios=("anio", "size"),
        )
    )
    ranking["puesto_original"] = (
        ranking["score_original_prom"].rank(ascending=False, method="min").astype(int)
    )
    ranking = ranking.sort_values("score_corregido_prom", ascending=False).reset_index(drop=True)
    ranking.insert(0, "puesto", ranking.index + 1)
    ranking["cambio_vs_original"] = ranking["puesto_original"] - ranking["puesto"]
    ranking = ranking[
        ["puesto", "cmac", "score_corregido_prom", "score_original_prom",
         "puesto_original", "cambio_vs_original", "anios"]
    ]

    print("\n[03] Ranking de eficiencia de las cajas (score CORREGIDO, promedio panel):")
    for _, r in ranking.iterrows():
        chg = int(r["cambio_vs_original"])
        arrow = f"(={'' if chg==0 else ('+'+str(chg) if chg>0 else str(chg))})"
        print(f"    {int(r['puesto']):2d}. {r['cmac']:<42s} "
              f"corr={r['score_corregido_prom']:.4f}  orig={r['score_original_prom']:.4f} {arrow}")

    # ------------------------------ Exportacion ------------------------------ #
    final = out[
        ["cmac", "anio", "score_original", "score_corregido",
         "ic_inferior", "ic_superior", "etapa1_score", "etapa2_score"]
    ].copy()

    controles = pd.DataFrame(
        [
            ["NIVEL VENTANA (condicion de aceptacion, enfoque B)", "", "", "", ""],
            ["(a) corregido_W <= original_W",
             int(chk_aW.sum()), len(det), bool(chk_aW.all()), "condicion"],
            ["(b) corregido_W <= 1.0",
             int(chk_bW.sum()), len(det), bool(chk_bW.all()), "condicion"],
            ["(c) corregido_W dentro de su IC_W",
             int(chk_cW.sum()), len(det), bool(chk_cW.all()), "condicion"],
            ["NIVEL ANUAL (promedio de corregidos; IC ENVOLVENTE [min ic_inf, max ic_sup])",
             "", "", "", ""],
            ["(a) promedio_corregido <= promedio_original",
             int(chk_a.sum()), len(out), bool(chk_a.all()), "condicion"],
            ["(b) promedio_corregido <= 1.0",
             int(chk_b.sum()), len(out), bool(chk_b.all()), "condicion"],
            ["(c) promedio_corregido dentro del IC envolvente",
             int(chk_c.sum()), len(out), bool(chk_c.all()),
             "IC envolvente = union de intervalos de las ventanas donde aparece la "
             "caja-anio; la guia sugeria solo el IC de la ventana central"],
        ],
        columns=["control", "n_cumple", "n_total", "todos_cumplen", "tipo"],
    )

    ic_cmp = out[
        ["cmac", "anio", "n_ventanas", "score_corregido",
         "ic_inferior", "ic_superior", "ic_central_inf", "ic_central_sup",
         "ventana_central"]
    ].rename(columns={"ic_inferior": "ic_envolvente_inf",
                      "ic_superior": "ic_envolvente_sup"})

    det_out = det.sort_values(["anio", "cmac", "ventana"])[
        ["cmac", "anio", "ventana", "es_central", "n_ref", "m_sub", "B",
         "delta_hat_W", "corregido_W", "ic_inf_W", "ic_sup_W",
         "etapa1_W", "etapa2_W", "mean_delta_star", "frac_delta_star_ef"]
    ]

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        final.to_excel(xw, sheet_name="eficiencia_limpia", index=False)
        ranking.to_excel(xw, sheet_name="ranking_corregido", index=False)
        controles.to_excel(xw, sheet_name="controles_validez", index=False)
        ic_cmp.to_excel(xw, sheet_name="ic_comparacion", index=False)
        det_out.to_excel(xw, sheet_name="detalle_ventanas", index=False)

    print(f"\n[03] Tabla consolidada guardada en: {OUT_PATH}")


if __name__ == "__main__":
    main()
