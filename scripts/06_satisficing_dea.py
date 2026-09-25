#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Capa 4 - Robustez estocastica: satisficing DEA
Guia Metodologica Definitiva (v4) - DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 06_satisficing_dea.py   (AUTOCONTENIDO Y REPLICABLE)

Ejecutable directo:  python scripts/06_satisficing_dea.py

QUE HACE
  Verifica que el ordenamiento de eficiencias del nucleo (Network SBM-DDF) es
  ESTABLE ante la volatilidad de los datos y no un artefacto determinista de una
  muestra pequena (N=12). Operacionaliza el satisficing/chance-constrained DEA
  de Charles, Tsolas & Gherman (2018) mediante SIMULACION MONTE CARLO:
    * Los insumos y productos se tratan como magnitudes sujetas a variabilidad
      estocastica (ruido multiplicativo lognormal; CV base 5%, sensibilidad 10%).
    * En cada replica se recomputa la eficiencia Network SBM-DDF (VRS, Kuosmanen)
      de cada caja-anio en su ventana central; se promedia por caja.
    * Para cada caja se estima la PROBABILIDAD DE SATISFACCION: P(eficiencia media
      >= nivel de aspiracion) para varios niveles, y la DISTRIBUCION de su rango.
    * Se contrasta el ranking probabilistico con el ranking DETERMINISTA del
      nucleo (Spearman y Kendall). Alta concordancia => diferencias robustas.

  [Nota honesta] Es la version por simulacion del concepto satisficing/chance-
  constrained; captura la probabilidad de alcanzar la aspiracion bajo
  incertidumbre de datos, no la forma cerrada del programa con restricciones de
  azar. Se declara en el reporte.

SALIDAS
  outputs/tables/satisficing_dea_capa4.xlsx
  outputs/figures/satisficing_ranking_capa4.png
================================================================================
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Libro1.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "tables"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
OUT_PATH = OUT_DIR / "satisficing_dea_capa4.xlsx"
FIG_PATH = FIG_DIR / "satisficing_ranking_capa4.png"

INPUTS = ["g_personal_r", "n_oficinas"]
INTERMEDIATE = "total_depositos_r"
GOOD_OUTPUTS = ["creditos_vigentes_r", "ingresos_finan_r"]
BAD_OUTPUT = "cartera_atrasada_r"
CORE = INPUTS + [INTERMEDIATE] + GOOD_OUTPUTS + [BAD_OUTPUT]

WINDOW = 3
B_REPS = 1000
SEED = 42
CV_LEVELS = [0.05, 0.10]                 # coeficientes de variacion (base, sensibilidad)
ASPIRATION = [0.75, 0.80, 0.85, 0.90]    # niveles de aspiracion de eficiencia


# ---------------- ventanas ---------------- #
def build_windows(years):
    ys = sorted(years); wins = []
    for s in range(ys[0], ys[-1] - WINDOW + 2):
        block = list(range(s, s + WINDOW))
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


# ---------------- Network SBM-DDF (identico a Capa 1) ---------------- #
def solve_nsbm_ddf(R, o):
    x1, x2 = R["x1"], R["x2"]; dep = R["dep"]; y1, y2, bb = R["y1"], R["y2"], R["b"]
    n = x1.shape[0]; nv = 3 * n + 5
    x1o, x2o = o["x1"], o["x2"]; y1o, y2o, bo = o["y1"], o["y2"], o["b"]
    i_sx1, i_sx2 = 3 * n, 3 * n + 1; i_sy1, i_sy2 = 3 * n + 2, 3 * n + 3; i_sb = 3 * n + 4
    c = np.zeros(nv)
    c[i_sx1] = -0.25 / x1o; c[i_sx2] = -0.25 / x2o
    c[i_sy1] = -(1 / 6) / y1o; c[i_sy2] = -(1 / 6) / y2o; c[i_sb] = -(1 / 6) / bo
    A_eq = np.zeros((7, nv)); b_eq = np.zeros(7)
    lam = slice(0, n); z2 = slice(n, 2 * n); u2 = slice(2 * n, 3 * n)
    A_eq[0, lam] = x1; A_eq[0, i_sx1] = 1; b_eq[0] = x1o
    A_eq[1, lam] = x2; A_eq[1, i_sx2] = 1; b_eq[1] = x2o
    A_eq[2, lam] = 1; b_eq[2] = 1
    A_eq[3, z2] = y1; A_eq[3, i_sy1] = -1; b_eq[3] = y1o
    A_eq[4, z2] = y2; A_eq[4, i_sy2] = -1; b_eq[4] = y2o
    A_eq[5, z2] = bb; A_eq[5, i_sb] = 1; b_eq[5] = bo
    A_eq[6, z2] = 1; A_eq[6, u2] = 1; b_eq[6] = 1
    A_ub = np.zeros((1, nv)); A_ub[0, lam] = -dep; A_ub[0, z2] = dep; A_ub[0, u2] = dep
    res = linprog(c, A_ub=A_ub, b_ub=[0.0], A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    v = res.x
    W1 = 0.5 * (v[i_sx1] / x1o + v[i_sx2] / x2o)
    W2 = (1 / 3) * (v[i_sy1] / y1o + v[i_sy2] / y2o + v[i_sb] / bo)
    return 1.0 / (1.0 + 0.5 * (W1 + W2))


# ---------------- scores por caja (central window) sobre una matriz de datos ---- #
def caja_scores(mat, years, anio, cmac_codes, cajas, cwmap):
    """mat: (N,6) en orden CORE. Devuelve dict caja-> media de sus scores centrales."""
    scores = {c: [] for c in cajas}
    for y in years:
        w = cwmap[y]
        ref_mask = np.isin(anio, w)
        ridx = np.where(ref_mask)[0]
        R = {"x1": mat[ridx, 0], "x2": mat[ridx, 1], "dep": mat[ridx, 2],
             "y1": mat[ridx, 3], "y2": mat[ridx, 4], "b": mat[ridx, 5]}
        oidx = np.where(anio == y)[0]
        for i in oidx:
            o = {"x1": mat[i, 0], "x2": mat[i, 1], "y1": mat[i, 3], "y2": mat[i, 4], "b": mat[i, 5]}
            s = solve_nsbm_ddf(R, o)
            scores[cmac_codes[i]].append(s)
    return {c: float(np.nanmean(v)) if v else np.nan for c, v in scores.items()}


# ---------------- worker Monte Carlo ---------------- #
def mc_worker(payload):
    draws, cv, base, years, anio, cmac_codes, cajas, cwmap = (
        payload["draws"], payload["cv"], payload["base"], payload["years"],
        payload["anio"], payload["cmac_codes"], payload["cajas"], payload["cwmap"])
    sig = cv
    out = np.full((len(draws), len(cajas)), np.nan)
    for k, b in enumerate(draws):
        rng = np.random.default_rng([SEED, int(round(cv * 1000)), int(b)])
        # ruido lognormal multiplicativo preservando la media
        eps = rng.normal(0, 1, base.shape)
        mat = base * np.exp(sig * eps - 0.5 * sig ** 2)
        sc = caja_scores(mat, years, anio, cmac_codes, cajas, cwmap)
        out[k] = [sc[c] for c in cajas]
    return out


def run_mc(cv, base, years, anio, cmac_codes, cajas, cwmap, n_workers):
    chunks = np.array_split(np.arange(B_REPS), n_workers)
    payloads = [{"draws": list(ch), "cv": cv, "base": base, "years": years, "anio": anio,
                 "cmac_codes": cmac_codes, "cajas": cajas, "cwmap": cwmap}
                for ch in chunks if len(ch)]
    parts = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(mc_worker, payloads):
            parts.append(r)
    return np.vstack(parts)   # (B, n_cajas)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[06] Cargando panel: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, sheet_name=0).sort_values(["anio", "cmac"]).reset_index(drop=True)
    years = sorted(df["anio"].unique())
    windows = build_windows(years)
    cwmap = {y: central_window_for_year(y, windows) for y in years}
    cajas = sorted(df["cmac"].unique())
    anio = df["anio"].to_numpy()
    cmac_codes = df["cmac"].to_numpy()
    base = df[CORE].to_numpy(float)

    # -------- ranking DETERMINISTA del nucleo (datos sin perturbar) -------- #
    det = caja_scores(base, years, anio, cmac_codes, cajas, cwmap)
    det_ser = pd.Series(det).sort_values(ascending=False)
    det_rank = {c: i + 1 for i, c in enumerate(det_ser.index)}
    print("[06] Ranking determinista del nucleo calculado.")

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"[06] Monte Carlo satisficing: B={B_REPS} x {len(CV_LEVELS)} CV, seed={SEED}, {n_workers} procesos")

    results = {}
    for cv in CV_LEVELS:
        print(f"     -> perturbacion CV={cv:.0%} ...")
        M = run_mc(cv, base, years, anio, cmac_codes, cajas, cwmap, n_workers)  # (B, ncaj)
        results[cv] = M

    # -------- metricas por CV -------- #
    writer_sheets = {}
    concord_rows = []
    for cv, M in results.items():
        meaneff = np.nanmean(M, axis=0)
        sdeff = np.nanstd(M, axis=0)
        # ranking por replica (mayor eficiencia = rango 1)
        order = (-M).argsort(axis=1).argsort(axis=1) + 1   # rank per draw
        mean_rank = order.mean(axis=0); sd_rank = order.std(axis=0)
        p_top3 = (order <= 3).mean(axis=0)
        p_tophalf = (order <= len(cajas) // 2).mean(axis=0)
        # probabilidades de satisfaccion
        sat = {f"P(ef>={a:.2f})": (M >= a).mean(axis=0) for a in ASPIRATION}
        prob_rank_order = np.argsort(-meaneff)
        tab = pd.DataFrame({
            "cmac": cajas,
            "ef_det": [det[c] for c in cajas],
            "ef_media_MC": meaneff, "ef_sd_MC": sdeff,
            "rango_det": [det_rank[c] for c in cajas],
            "rango_medio_MC": mean_rank, "rango_sd_MC": sd_rank,
            "P_top3": p_top3, "P_top_mitad": p_tophalf,
            **sat,
        }).sort_values("ef_media_MC", ascending=False).reset_index(drop=True)
        tab.insert(0, "rango_prob", tab.index + 1)
        writer_sheets[f"cv_{int(cv*100)}pct"] = tab
        # concordancia determinista vs probabilistico (por eficiencia media)
        det_vec = np.array([det[c] for c in cajas])
        rho, prho = spearmanr(det_vec, meaneff)
        tau, ptau = kendalltau(det_vec, meaneff)
        # % de cajas que conservan su rango exacto y +-1
        prob_rank = {c: r for c, r in zip(tab["cmac"], tab["rango_prob"])}
        same = np.mean([det_rank[c] == prob_rank[c] for c in cajas])
        within1 = np.mean([abs(det_rank[c] - prob_rank[c]) <= 1 for c in cajas])
        concord_rows.append({"CV": f"{int(cv*100)}%", "spearman": rho, "spearman_p": prho,
                             "kendall_tau": tau, "kendall_p": ptau,
                             "rango_identico_%": 100 * same, "rango_+-1_%": 100 * within1})
        print(f"     CV={cv:.0%}: Spearman={rho:.3f} (p={prho:.3g}) | Kendall={tau:.3f} | "
              f"rango identico={100*same:.0f}% | +-1={100*within1:.0f}%")

    concord = pd.DataFrame(concord_rows)

    # -------- figura (distribucion de eficiencia por caja, CV base) -------- #
    make_figure(results[CV_LEVELS[0]], cajas, det, det_rank, CV_LEVELS[0])
    print(f"[06] Figura: {FIG_PATH}")

    # -------- exportacion -------- #
    resumen = pd.DataFrame([
        {"parametro": "Modelo", "valor": "Network SBM-DDF (nucleo, VRS, Kuosmanen)"},
        {"parametro": "Perturbacion", "valor": "lognormal multiplicativa (preserva media)"},
        {"parametro": "CV evaluados", "valor": ", ".join(f"{int(c*100)}%" for c in CV_LEVELS)},
        {"parametro": "Replicas", "valor": f"B={B_REPS} por CV, seed={SEED}"},
        {"parametro": "Niveles de aspiracion", "valor": ", ".join(f"{a:.2f}" for a in ASPIRATION)},
        {"parametro": "Concordancia", "valor": "Spearman/Kendall (determinista vs probabilistico)"},
    ])
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="resumen", index=False)
        concord.to_excel(xw, sheet_name="concordancia", index=False)
        for name, tab in writer_sheets.items():
            tab.round(4).to_excel(xw, sheet_name=name, index=False)
    print(f"[06] Resultados: {OUT_PATH}")


def make_figure(M, cajas, det, det_rank, cv):
    order = sorted(range(len(cajas)), key=lambda i: det[cajas[i]], reverse=True)
    data = [M[:, i][np.isfinite(M[:, i])] for i in order]
    labels = [cajas[i].replace("Caja Municipal de Crédito Popular Lima", "CMCP Lima") for i in order]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.62, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dfe6f0"); patch.set_edgecolor("#2f4b7c")
    for med in bp["medians"]:
        med.set_color("#a5460f"); med.set_linewidth(1.6)
    for i, idx in enumerate(order):
        ax.plot(det[cajas[idx]], i + 1, "D", color="#136f63", ms=6, zorder=5,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Eficiencia Network SBM-DDF bajo perturbación estocástica")
    ax.set_title(f"Capa 4 — Satisficing DEA: distribución de eficiencia por caja (CV={int(cv*100)}%, B={M.shape[0]})",
                 fontsize=11.5, weight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="D", color="w", markerfacecolor="#136f63",
                              markersize=7, label="Eficiencia determinista"),
                       Line2D([0], [0], color="#a5460f", lw=1.6, label="Mediana Monte Carlo")],
              loc="lower right", fontsize=8.5)
    ax.grid(True, axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG_PATH, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
