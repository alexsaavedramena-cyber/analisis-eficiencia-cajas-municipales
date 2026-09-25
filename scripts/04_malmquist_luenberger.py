#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Capa 2 - Dinamica de productividad: indice Malmquist-Luenberger (ML)
Guia Metodologica Definitiva (v4) - DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 04_malmquist_luenberger.py   (AUTOCONTENIDO Y REPLICABLE)

Ejecutable directo:  python scripts/04_malmquist_luenberger.py
No depende de ningun otro script: carga Libro1.xlsx, define sus propias rutinas
de optimizacion lineal (scipy/HiGHS), calcula el indice y exporta resultados.

--------------------------------------------------------------------------------
QUE HACE
--------------------------------------------------------------------------------
Estima el indice Malmquist-Luenberger (Chung, Fare & Grosskopf, 1997) sobre
FRONTERAS ANUALES INDEPENDIENTES y lo descompone en:
  * EC  (Efficiency Change / catch-up): cambio en la eficiencia relativa.
  * TC  (Technological Change): desplazamiento de la frontera.
con  ML = EC x TC.

ESPECIFICACION PARSIMONIOSA REDUCIDA  -  Contraste C (m+s=3):
  Con solo 11-12 DMU por anio, el nucleo de 6 variables viola la regla de Cooper
  en fronteras contemporaneas. Se usa la especificacion reducida que INTERNALIZA
  el bad output:
      insumo (x)        : total_depositos_r      (fondos intermediados)
      producto bueno (y): creditos_vigentes_r    (colocacion sana)
      producto malo (b) : cartera_atrasada_r     (riesgo de credito realizado)
  Mide la productividad de la COLOCACION ajustada por riesgo. m+s = 1+2 = 3 <= N/3.

FUNCION DE DISTANCIA DIRECCIONAL (DDF) - Chung-Fare-Grosskopf, con bad output:
  D(x,y,b) = max beta   s.a.   Sum_j lam_j x_j <= x_o           (insumo)
                                Sum_j lam_j y_j >= (1+beta) y_o  (bueno, fuerte disp.)
                                Sum_j lam_j b_j  = (1-beta) b_o  (malo, disp. DEBIL)
                                lam_j >= 0
  Direccion g = (y_o, b_o): expandir el bueno y contraer el malo (ajuste por
  riesgo). Tecnologia CRS (estandar en productividad; radialmente tratable, lo que
  habilita el bootstrap de Simar & Wilson). beta = D es la ineficiencia dirigida.

INDICE Y DESCOMPOSICION (D^p(s) = distancia de la obs del anio s a la frontera p):
  ML   = [ (1+D^t(t))/(1+D^t(t+1)) * (1+D^{t+1}(t))/(1+D^{t+1}(t+1)) ]^(1/2)
  EC   = (1+D^t(t)) / (1+D^{t+1}(t+1))
  TC   = [ (1+D^{t+1}(t))/(1+D^t(t)) * (1+D^{t+1}(t+1))/(1+D^t(t+1)) ]^(1/2)
  ML>1: ganancia de productividad ; EC>1: catch-up (mejora de gestion) ;
  TC>1: progreso tecnologico (la frontera se desplaza hacia afuera).

BOOTSTRAP (Simar & Wilson, 1999) - IC 95%, B=2000, seed=42:
  Bootstrap suavizado homogeneo de las ineficiencias por frontera anual: se
  remuestrean con reemplazo y se suavizan con kernel gaussiano (regla de
  Silverman) con correccion de varianza y reflexion en 0; se reconstruye una
  pseudo-frontera proyectando cada DMU y se recomputan las cuatro distancias y el
  ML. Se repite B veces y se toman percentiles 2.5/97.5. [Nota honesta: se usa la
  variante homogenea por frontera; la version bivariante completa de S&W modela
  ademas la dependencia temporal entre periodos.]

AGREGACION POR SUBPERIODOS (media geometrica, propia de indices de productividad),
por anio final t+1 de cada transicion:
  Expansion 2002-2013 | Saturacion 2014-2019 | Choque/Poscrisis 2020-2025
Se evalua el patron "gestion al alza (EC>1) vs tecnologia a la baja (TC<1)".

--------------------------------------------------------------------------------
SALIDAS
--------------------------------------------------------------------------------
  outputs/tables/malmquist_luenberger.xlsx
  outputs/figures/evolucion_ml_subperiodos.png
================================================================================
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------ #
# Configuracion
# ------------------------------------------------------------------ #
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Libro1.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "tables"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"
OUT_PATH = OUT_DIR / "malmquist_luenberger.xlsx"
FIG_PATH = FIG_DIR / "evolucion_ml_subperiodos.png"

# Especificacion reducida C (m+s=3), bad output internalizado
X_VAR = "total_depositos_r"       # insumo
YG_VAR = "creditos_vigentes_r"    # producto bueno
YB_VAR = "cartera_atrasada_r"     # producto malo (bad output)

CRS = True                        # tecnologia de retornos constantes
B_REPS = 2000
SEED = 42
ALPHA = 0.05

SUBPERIODS = {
    "Expansion (2002-2013)": (2002, 2013),
    "Saturacion (2014-2019)": (2014, 2019),
    "Choque/Poscrisis (2020-2025)": (2020, 2025),
}


# ================================================================== #
# Funcion de distancia direccional (DDF) con bad output  - HiGHS
# ================================================================== #
def ddf(xr, ygr, ybr, xo, ygo, ybo, crs=True):
    """
    Distancia direccional D(x_o,y_o,b_o) contra la frontera dada por
    (xr, ygr, ybr). Direccion g=(ygo, ybo). Devuelve beta=D o np.nan si
    infactible/ilimitado.
      variables: v = [lam_1..lam_n, beta]
      max beta  ==  min -beta
    """
    n = xr.shape[0]
    nv = n + 1
    c = np.zeros(nv); c[-1] = -1.0

    A_ub = []; b_ub = []
    # insumo:  sum lam_j x_j <= x_o
    row = np.zeros(nv); row[:n] = xr; row[-1] = 0.0
    A_ub.append(row); b_ub.append(xo)
    # producto bueno: sum lam_j y_j >= (1+beta) y_o  ->  -sum lam y + beta*y_o <= -y_o
    row = np.zeros(nv); row[:n] = -ygr; row[-1] = ygo
    A_ub.append(row); b_ub.append(-ygo)

    # producto malo (igualdad, disp. debil): sum lam_j b_j + beta*b_o = b_o
    A_eq = []; b_eq = []
    row = np.zeros(nv); row[:n] = ybr; row[-1] = ybo
    A_eq.append(row); b_eq.append(ybo)
    # VRS opcional
    if not crs:
        row = np.zeros(nv); row[:n] = 1.0; row[-1] = 0.0
        A_eq.append(row); b_eq.append(1.0)

    bounds = [(0, None)] * n + [(None, None)]   # lam>=0 ; beta libre
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if not res.success:
        return np.nan
    return float(res.x[-1])


# ================================================================== #
# Carga y organizacion del panel por anio
# ================================================================== #
def load_year_data():
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    years = sorted(df["anio"].unique())
    data = {}
    for y in years:
        g = df[df["anio"] == y].set_index("cmac").sort_index()
        data[y] = {
            "cmac": list(g.index),
            "x": g[X_VAR].to_numpy(float),
            "yg": g[YG_VAR].to_numpy(float),
            "yb": g[YB_VAR].to_numpy(float),
        }
    return df, years, data


def subperiod_of(t1):
    for name, (a, b) in SUBPERIODS.items():
        if a <= t1 <= b:
            return name
    return None


# ================================================================== #
# Estimaciones puntuales del ML por caja y transicion
# ================================================================== #
def point_estimates(years, data):
    rows = []
    for t in years[:-1]:
        t1 = t + 1
        Dt = data[t]; Dt1 = data[t1]
        common = [c for c in Dt["cmac"] if c in set(Dt1["cmac"])]
        # arrays de referencia
        for c in common:
            it = Dt["cmac"].index(c); it1 = Dt1["cmac"].index(c)
            zt = (Dt["x"][it], Dt["yg"][it], Dt["yb"][it])
            zt1 = (Dt1["x"][it1], Dt1["yg"][it1], Dt1["yb"][it1])
            # cuatro distancias
            Dtt = ddf(Dt["x"], Dt["yg"], Dt["yb"], *zt, crs=CRS)              # D^t(t)
            Dt1t1 = ddf(Dt1["x"], Dt1["yg"], Dt1["yb"], *zt1, crs=CRS)        # D^{t+1}(t+1)
            Dtt1 = ddf(Dt["x"], Dt["yg"], Dt["yb"], *zt1, crs=CRS)            # D^t(t+1)
            Dt1t = ddf(Dt1["x"], Dt1["yg"], Dt1["yb"], *zt, crs=CRS)          # D^{t+1}(t)
            ml, ec, tc = ml_components(Dtt, Dt1t1, Dtt1, Dt1t)
            rows.append({
                "cmac": c, "t": t, "t1": t1,
                "subperiodo": subperiod_of(t1),
                "D_t_t": Dtt, "D_t1_t1": Dt1t1, "D_t_t1": Dtt1, "D_t1_t": Dt1t,
                "ML": ml, "EC": ec, "TC": tc,
            })
    return pd.DataFrame(rows)


def ml_components(Dtt, Dt1t1, Dtt1, Dt1t):
    """Devuelve (ML, EC, TC) con guardas de dominio; np.nan si algun (1+D)<=0."""
    a = 1 + Dtt; b = 1 + Dt1t1; c = 1 + Dt1t; d = 1 + Dtt1
    if min(a, b, c, d) <= 1e-9 or any(np.isnan([a, b, c, d])):
        return np.nan, np.nan, np.nan
    ec = a / b
    tc = np.sqrt((c / a) * (b / d))
    ml = np.sqrt((a / d) * (c / b))
    return ml, ec, tc


def geomean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(v))))


# ================================================================== #
# Bootstrap suavizado (Simar & Wilson, 1999) por frontera anual
# ================================================================== #
def precompute_bootstrap(years, data):
    """Por anio: ineficiencias propias D^t(t), proyecciones a frontera, y stats."""
    pre = {}
    for y in years:
        D = data[y]; n = len(D["cmac"])
        delta = np.array([
            ddf(D["x"], D["yg"], D["yb"], D["x"][i], D["yg"][i], D["yb"][i], crs=CRS)
            for i in range(n)
        ], float)
        delta = np.clip(np.nan_to_num(delta, nan=0.0), 0.0, 0.999)
        # proyeccion a la frontera (punto eficiente en direccion g)
        yg_front = D["yg"] * (1 + delta)
        yb_front = D["yb"] * (1 - delta)
        sd = np.std(delta) if np.std(delta) > 1e-8 else 1e-8
        h = 1.06 * sd * (n ** (-1 / 5))     # regla de Silverman
        pre[y] = {
            "x": D["x"], "yg": D["yg"], "yb": D["yb"],
            "yg_front": yg_front, "yb_front": yb_front,
            "delta": delta, "mean": float(delta.mean()), "var": float(sd ** 2),
            "h": float(h), "cmac": D["cmac"], "n": n,
        }
    return pre


def smoothed_scores(pre_y, rng):
    """Genera pseudo-ineficiencias suavizadas (S&W) y devuelve pseudo-productos."""
    delta = pre_y["delta"]; n = pre_y["n"]; h = pre_y["h"]
    m = pre_y["mean"]; var = pre_y["var"]
    idx = rng.integers(0, n, n)
    beta = delta[idx]
    eps = rng.normal(0, 1, n)
    dt = beta + h * eps
    dt = np.where(dt < 0, -dt, dt)                        # reflexion en 0
    dstar = m + (dt - m) / np.sqrt(1 + h * h / max(var, 1e-8))  # correc. varianza
    dstar = np.clip(np.abs(dstar), 0.0, 0.999)
    yg_star = pre_y["yg_front"] / (1 + dstar)
    yb_star = pre_y["yb_front"] / (1 - dstar)
    return yg_star, yb_star


def boot_worker(args):
    """Ejecuta un bloque de replicas b y devuelve agregados por subperiodo/anio."""
    b_list, years, pre, transitions = args
    sub_names = list(SUBPERIODS.keys())
    out = {"sub": {s: {"ML": [], "EC": [], "TC": []} for s in sub_names},
           "ann": {t1: {"ML": [], "EC": [], "TC": []} for (_, t1) in transitions}}
    for b in b_list:
        rng = np.random.default_rng([SEED, int(b)])
        # pseudo-frontera por anio
        star = {}
        for y in years:
            ygs, ybs = smoothed_scores(pre[y], rng)
            star[y] = (pre[y]["x"], ygs, ybs)
        # ML por transicion
        per_sub = {s: {"ML": [], "EC": [], "TC": []} for s in sub_names}
        for (t, t1) in transitions:
            xt, ygt, ybt = star[t]; xt1, ygt1, ybt1 = star[t1]
            preT = pre[t]; preT1 = pre[t1]
            common = [c for c in preT["cmac"] if c in set(preT1["cmac"])]
            mls, ecs, tcs = [], [], []
            for c in common:
                it = preT["cmac"].index(c); it1 = preT1["cmac"].index(c)
                zt = (preT["x"][it], preT["yg"][it], preT["yb"][it])
                zt1 = (preT1["x"][it1], preT1["yg"][it1], preT1["yb"][it1])
                Dtt = ddf(xt, ygt, ybt, *zt, crs=CRS)
                Dt1t1 = ddf(xt1, ygt1, ybt1, *zt1, crs=CRS)
                Dtt1 = ddf(xt, ygt, ybt, *zt1, crs=CRS)
                Dt1t = ddf(xt1, ygt1, ybt1, *zt, crs=CRS)
                ml, ec, tc = ml_components(Dtt, Dt1t1, Dtt1, Dt1t)
                mls.append(ml); ecs.append(ec); tcs.append(tc)
            s = subperiod_of(t1)
            out["ann"][t1]["ML"].append(geomean(mls))
            out["ann"][t1]["EC"].append(geomean(ecs))
            out["ann"][t1]["TC"].append(geomean(tcs))
            per_sub[s]["ML"] += mls; per_sub[s]["EC"] += ecs; per_sub[s]["TC"] += tcs
        for s in sub_names:
            out["sub"][s]["ML"].append(geomean(per_sub[s]["ML"]))
            out["sub"][s]["EC"].append(geomean(per_sub[s]["EC"]))
            out["sub"][s]["TC"].append(geomean(per_sub[s]["TC"]))
    return out


def run_bootstrap(years, data, transitions):
    pre = precompute_bootstrap(years, data)
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    chunks = np.array_split(np.arange(B_REPS), n_workers)
    args = [(list(ch), years, pre, transitions) for ch in chunks if len(ch)]
    print(f"[04] Bootstrap S&W: B={B_REPS}, seed={SEED}, {n_workers} procesos...")
    sub_names = list(SUBPERIODS.keys())
    agg = {"sub": {s: {"ML": [], "EC": [], "TC": []} for s in sub_names},
           "ann": {t1: {"ML": [], "EC": [], "TC": []} for (_, t1) in transitions}}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(boot_worker, args):
            for s in sub_names:
                for k in ("ML", "EC", "TC"):
                    agg["sub"][s][k] += res["sub"][s][k]
            for (_, t1) in transitions:
                for k in ("ML", "EC", "TC"):
                    agg["ann"][t1][k] += res["ann"][t1][k]
    return agg


def ci(dist):
    d = np.asarray(dist, float); d = d[np.isfinite(d)]
    if d.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(d, ALPHA / 2)), float(np.quantile(d, 1 - ALPHA / 2)))


# ================================================================== #
# Figura
# ================================================================== #
def make_figure(annual_df):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = {"Expansion (2002-2013)": "#dfe9f3",
              "Saturacion (2014-2019)": "#f3ecdf",
              "Choque/Poscrisis (2020-2025)": "#f3dfe0"}
    for name, (a, b) in SUBPERIODS.items():
        ax.axvspan(a - 0.5, b + 0.5, color=colors[name], alpha=.7, zorder=0)
    x = annual_df["t1"].to_numpy()
    ax.plot(x, annual_df["ML"], "-o", color="#2f4b7c", lw=2.2, ms=5, label="ML (productividad)", zorder=4)
    ax.fill_between(x, annual_df["ML_ic_inf"], annual_df["ML_ic_sup"],
                    color="#2f4b7c", alpha=.15, zorder=1, label="IC 95% de ML")
    ax.plot(x, annual_df["EC"], "--s", color="#1f7a5a", lw=1.8, ms=4, label="EC (catch-up / gestion)", zorder=3)
    ax.plot(x, annual_df["TC"], "--^", color="#a5460f", lw=1.8, ms=4, label="TC (cambio tecnologico)", zorder=3)
    ax.axhline(1.0, color="#444", lw=1, ls=":", zorder=2)
    ax.set_xlabel("Año final de la transición (t+1)")
    ax.set_ylabel("Índice (geométrico anual)")
    ax.set_title("Malmquist-Luenberger de las CMAC: productividad, catch-up y cambio tecnológico",
                 fontsize=12, weight="bold")
    handles = ax.get_legend_handles_labels()[0]
    handles += [Patch(facecolor=colors[k], label=k) for k in SUBPERIODS]
    ax.legend(handles=handles, loc="best", fontsize=8.5, ncol=2, framealpha=.9)
    ax.grid(True, axis="y", alpha=.25)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


# ================================================================== #
# Main
# ================================================================== #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[04] Cargando panel: {DATA_PATH}")
    df, years, data = load_year_data()
    transitions = [(t, t + 1) for t in years[:-1]]
    print(f"[04] Especificacion C: x={X_VAR} -> y={YG_VAR} (+), b={YB_VAR} (-) | "
          f"tecnologia={'CRS' if CRS else 'VRS'}")
    print(f"[04] {len(transitions)} transiciones anuales ({years[0]}-{years[-1]})")

    # 1) Estimaciones puntuales
    pe = point_estimates(years, data)
    n_nan = int(pe["ML"].isna().sum())
    print(f"[04] ML por caja-transicion: {len(pe)} | no calculables: {n_nan}")

    # 2) Agregados puntuales por subperiodo (media geometrica)
    sub_rows = []
    for s in SUBPERIODS:
        sub = pe[pe["subperiodo"] == s]
        n_val = int(np.isfinite(sub["ML"]).sum())
        sub_rows.append({"subperiodo": s,
                         "ML": geomean(sub["ML"]), "EC": geomean(sub["EC"]),
                         "TC": geomean(sub["TC"]),
                         "n_validas": n_val, "n_total": len(sub)})
    sub_point = pd.DataFrame(sub_rows)

    # 3) Bootstrap S&W -> IC
    agg = run_bootstrap(years, data, transitions)

    for r in sub_rows:
        s = r["subperiodo"]
        for k in ("ML", "EC", "TC"):
            lo, hi = ci(agg["sub"][s][k])
            r[f"{k}_ic_inf"] = lo; r[f"{k}_ic_sup"] = hi
    sub_point = pd.DataFrame(sub_rows)
    sub_point["patron_EC>1_TC<1"] = (
        (sub_point["EC"] > 1) & (sub_point["TC"] < 1)
    )

    # 4) Serie anual (media geometrica por transicion) + IC de ML
    ann_rows = []
    for (t, t1) in transitions:
        sub = pe[pe["t1"] == t1]
        row = {"t": t, "t1": t1, "subperiodo": subperiod_of(t1),
               "ML": geomean(sub["ML"]), "EC": geomean(sub["EC"]), "TC": geomean(sub["TC"])}
        for k in ("ML", "EC", "TC"):
            lo, hi = ci(agg["ann"][t1][k])
            row[f"{k}_ic_inf"] = lo; row[f"{k}_ic_sup"] = hi
        ann_rows.append(row)
    annual_df = pd.DataFrame(ann_rows)

    # 5) Hipotesis global
    print("\n[04] Patron gestion-al-alza (EC>1) vs tecnologia-a-la-baja (TC<1):")
    for _, r in sub_point.iterrows():
        flag = "SI CONFIRMA" if r["patron_EC>1_TC<1"] else "no confirma"
        print(f"    {r['subperiodo']:<32s} ML={r['ML']:.3f}  EC={r['EC']:.3f}  "
              f"TC={r['TC']:.3f}  -> {flag}")

    # 6) Figura
    make_figure(annual_df)
    print(f"[04] Figura guardada: {FIG_PATH}")

    # 7) Exportacion
    resumen = pd.DataFrame([{
        "parametro": "Especificacion", "valor": "C (m+s=3): depositos -> vigentes(+), atrasada(-)"},
        {"parametro": "Tecnologia", "valor": "CRS"},
        {"parametro": "Direccion g", "valor": "(y_o, b_o): expandir bueno, contraer malo"},
        {"parametro": "Transiciones", "valor": f"{len(transitions)} ({years[0]}-{years[-1]})"},
        {"parametro": "Bootstrap", "valor": f"Simar & Wilson (1999) suavizado, B={B_REPS}, seed={SEED}"},
        {"parametro": "IC", "valor": "percentiles 2.5 / 97.5"},
    ])
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="resumen", index=False)
        sub_point.to_excel(xw, sheet_name="subperiodos", index=False)
        annual_df.to_excel(xw, sheet_name="serie_anual", index=False)
        pe.sort_values(["t1", "cmac"]).to_excel(xw, sheet_name="detalle_caja_transicion", index=False)
    print(f"[04] Resultados guardados: {OUT_PATH}")


if __name__ == "__main__":
    main()
