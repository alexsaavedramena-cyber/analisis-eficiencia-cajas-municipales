#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Capa 3 - Inferencia contextual: regresion truncada con doble bootstrap (panel)
Guia Metodologica Definitiva (v4) - DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 05_truncated_regression.py   (AUTOCONTENIDO Y REPLICABLE)

Ejecutable directo:  python scripts/05_truncated_regression.py

QUE HACE
  Segunda etapa del DEA: explica la eficiencia corregida (score_corregido, de la
  Capa 1) en funcion de un vector Z de variables contextuales, con inferencia
  valida ante la dependencia entre puntajes.

  Lee directamente 'panel etapa2.xlsx' (286 obs caja-anio).
  Variable dependiente: score_corregido  (delta_corr en (0,1]).
  Vector Z (candidatos): roe, roa, ratio_creditos_depositos,
    apalancamiento_pasivo_capital_reservas, total_activo_r (-> ln), dolarizacion,
    pbi_crec_real, ipc_va, d_pandemia, d_fintech.

PASOS
  A. Analisis estadistico previo (descriptivos, asimetria/curtosis, correlaciones, VIF).
  B. Criterio de seleccion de variables (VIF por pasos, umbral 5) + log del activo.
  C. Test de separabilidad (Daraio, Simar & Wilson, 2018) - implementacion practica.
  D. Regresion truncada con DOBLE BOOTSTRAP de panel (Simar & Wilson, 2007, Alg. II;
     Du et al., 2018): B1=1000 (correccion de sesgo de beta), B2=2000 (inferencia).
     Ademas: IC cluster-robusto por caja (dependencia de panel).
  E. Tobit (censura superior en 1) de referencia, con su invalidez inferencial.

NOTAS DE RIGOR
  * La eficiencia ya viene corregida por sesgo (KSW, Capa 1); por tanto el primer
    bootstrap de la Alg. II (correccion del sesgo DEA re-estimando fronteras) esta
    cumplido aguas arriba. Aqui el doble bootstrap opera sobre la REGRESION:
    B1 corrige el sesgo de los coeficientes beta (bootstrap parametrico de residuos
    truncados) y B2 construye IC/p-valores; se agrega un bootstrap por CONGLOMERADO
    de caja para respetar la dependencia serial de panel (Du et al., 2018).
  * Truncacion: score_corregido en (0,1] con eficientes en el limite SUPERIOR 1,
    de modo que la normal se trunca por la derecha en 1 (solo 4/286 en el borde).
  * El test de separabilidad se implementa en su version practica (asociacion
    Z-eficiencia con nulo por permutacion); se declara su alcance.

SALIDAS
  outputs/tables/regresion_truncada_capa3.xlsx
  outputs/reporte_capa3_estimacion.txt   (paso a paso legible)
================================================================================
"""

from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ------------------------------------------------------------------ #
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "tables" / "panel etapa2.xlsx"
OUT_XLSX = PROJECT_DIR / "outputs" / "tables" / "regresion_truncada_capa3.xlsx"
OUT_TXT = PROJECT_DIR / "outputs" / "reporte_capa3_estimacion.txt"

DEP = "score_corregido"
ID, TIME = "cmac", "anio"
CONT = ["roe", "roa", "ratio_creditos_depositos",
        "apalancamiento_pasivo_capital_reservas", "ln_activo",
        "dolarizacion_depositos", "pbi_crec_real", "ipc_va"]
DUMMIES = ["d_pandemia", "d_fintech"]
UPPER = 1.0
B1, B2 = 1000, 2000
SEED = 42
VIF_THRESHOLD = 5.0

# Signos esperados (para la discusion economica)
EXPECTED = {
    "const": None,
    "roe": "+", "roa": "+",
    "ratio_creditos_depositos": "+",
    "apalancamiento_pasivo_capital_reservas": "-",
    "ln_activo": "+",
    "dolarizacion_depositos": "-",
    "pbi_crec_real": "+",
    "ipc_va": "-",
    "d_pandemia": "-",
    "d_fintech": "-",
}


# ================================================================== #
# MLE: regresion truncada (normal truncada por la derecha en UPPER)
# ================================================================== #
def _neg_ll_trunc(theta, y, X, upper):
    k = X.shape[1]
    beta = theta[:k]; sigma = np.exp(theta[k])
    mu = X @ beta
    z = (y - mu) / sigma
    logpdf = stats.norm.logpdf(z) - np.log(sigma)
    logcdf = stats.norm.logcdf((upper - mu) / sigma)
    ll = logpdf - logcdf
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -np.sum(ll)


def _nll_and_grad(theta, y, X, upper):
    """Negativo de la log-verosimilitud truncada (derecha en 'upper') y su
    gradiente ANALITICO. Reparametriza sigma = exp(tau). lambda = phi/Phi (Mills)."""
    k = X.shape[1]
    beta = theta[:k]; tau = theta[k]; sigma = np.exp(tau)
    mu = X @ beta
    z = (y - mu) / sigma
    c = (upper - mu) / sigma
    logPhi = stats.norm.logcdf(c)
    lam = np.exp(stats.norm.logpdf(c) - logPhi)          # razon inversa de Mills
    ll = -0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - tau - logPhi
    if not np.all(np.isfinite(ll)):
        return 1e10, np.zeros(k + 1)
    nll = -np.sum(ll)
    grad_beta = -(X.T @ (z + lam)) / sigma               # d(nll)/d(beta)
    grad_tau = -np.sum(z ** 2 - 1.0 + c * lam)           # d(nll)/d(tau)
    return nll, np.concatenate([grad_beta, [grad_tau]])


def fit_truncated(y, X, upper=UPPER):
    k = X.shape[1]
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    s0 = max(np.std(y - X @ b0), 1e-3)
    theta0 = np.concatenate([b0, [np.log(s0)]])
    res = minimize(_nll_and_grad, theta0, args=(y, X, upper), jac=True,
                   method="L-BFGS-B", options={"maxiter": 1000})
    if not res.success:
        res = minimize(_nll_and_grad, theta0, args=(y, X, upper), jac=True,
                       method="BFGS", options={"maxiter": 2000})
    beta = res.x[:k]; sigma = np.exp(res.x[k])
    return beta, sigma, -res.fun


def truncated_se_hessian(beta, sigma, y, X, upper=UPPER):
    """SE asintotico via Hessiano numerico (referencia; la inferencia es bootstrap)."""
    k = X.shape[1]
    theta = np.concatenate([beta, [np.log(sigma)]])
    eps = 1e-5
    H = np.zeros((k + 1, k + 1))
    g0 = _grad(theta, y, X, upper, eps)
    for i in range(k + 1):
        tp = theta.copy(); tp[i] += eps
        gp = _grad(tp, y, X, upper, eps)
        H[:, i] = (gp - g0) / eps
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))[:k]
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return se


def _grad(theta, y, X, upper, eps):
    n = len(theta); g = np.zeros(n)
    f0 = _neg_ll_trunc(theta, y, X, upper)
    for i in range(n):
        tp = theta.copy(); tp[i] += eps
        g[i] = (_neg_ll_trunc(tp, y, X, upper) - f0) / eps
    return g


def draw_truncated(mu, sigma, upper, rng):
    """Muestra de N(mu_i, sigma) truncada por la derecha en 'upper' (vectorizado)."""
    b = (upper - mu) / sigma
    a = np.full_like(mu, -np.inf)
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng)


# ================================================================== #
# MLE: Tobit con censura superior en UPPER (referencia)
# ================================================================== #
def _neg_ll_tobit(theta, y, X, upper):
    k = X.shape[1]
    beta = theta[:k]; sigma = np.exp(theta[k])
    mu = X @ beta
    cens = y >= upper - 1e-9
    z = (y - mu) / sigma
    ll = np.empty(len(y))
    ll[~cens] = stats.norm.logpdf(z[~cens]) - np.log(sigma)
    ll[cens] = stats.norm.logsf((upper - mu[cens]) / sigma)   # P(y* >= upper)
    if not np.all(np.isfinite(ll)):
        return 1e10
    return -np.sum(ll)


def fit_tobit(y, X, upper=UPPER):
    k = X.shape[1]
    b0, *_ = np.linalg.lstsq(X, y, rcond=None)
    s0 = max(np.std(y - X @ b0), 1e-3)
    theta0 = np.concatenate([b0, [np.log(s0)]])
    res = minimize(_neg_ll_tobit, theta0, args=(y, X, upper), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    beta = res.x[:k]; sigma = np.exp(res.x[k])
    # SE via Hessiano
    eps = 1e-5; H = np.zeros((k + 1, k + 1))
    def gnll(t):
        gg = np.zeros(k + 1); f0 = _neg_ll_tobit(t, y, X, upper)
        for i in range(k + 1):
            tp = t.copy(); tp[i] += eps; gg[i] = (_neg_ll_tobit(tp, y, X, upper) - f0) / eps
        return gg
    g0 = gnll(res.x)
    for i in range(k + 1):
        tp = res.x.copy(); tp[i] += eps; H[:, i] = (gnll(tp) - g0) / eps
    H = 0.5 * (H + H.T)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))[:k]
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return beta, sigma, se


# ================================================================== #
# VIF y seleccion de variables por pasos
# ================================================================== #
def vif_series(df):
    out = {}
    cols = list(df.columns)
    for c in cols:
        y = df[c].values
        Xo = df.drop(columns=[c]).values
        Xo = np.column_stack([np.ones(len(Xo)), Xo])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r2 = 1 - ((y - Xo @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def stepwise_vif(df_cont, threshold):
    """Elimina iterativamente la variable con mayor VIF hasta que todas <= umbral."""
    keep = list(df_cont.columns); dropped = []
    while len(keep) > 1:
        v = vif_series(df_cont[keep])
        vmax = v.max()
        if vmax <= threshold:
            break
        drop = v.idxmax()
        dropped.append((drop, round(float(vmax), 3)))
        keep.remove(drop)
    return keep, dropped, vif_series(df_cont[keep])


# ================================================================== #
# Test de separabilidad (Daraio, Simar & Wilson, 2018) - version practica
# ================================================================== #
def separability_test(scores, Zc, rng, n_perm=2000):
    """
    Version practica: indice Z (1er componente principal estandarizado); particion
    por la mediana; estadistico = diferencia estandarizada de la eficiencia media
    entre grupos; nulo por permutacion de etiquetas. Un rechazo indica asociacion
    Z-eficiencia (condicion necesaria para la 2a etapa). El contraste completo
    frontera-vs-distribucion requiere estimar fronteras condicionales (se declara).
    """
    Zs = (Zc - Zc.mean(0)) / Zc.std(0)
    # 1er componente principal
    u, s, vt = np.linalg.svd(Zs, full_matrices=False)
    idx = u[:, 0]
    grp = idx >= np.median(idx)
    m1, m0 = scores[grp].mean(), scores[~grp].mean()
    sp = np.sqrt(scores[grp].var(ddof=1) / grp.sum() + scores[~grp].var(ddof=1) / (~grp).sum())
    stat = (m1 - m0) / sp
    # nulo por permutacion
    perm = np.empty(n_perm)
    y = np.asarray(scores)
    for i in range(n_perm):
        pg = rng.permutation(grp)
        perm[i] = (y[pg].mean() - y[~pg].mean()) / sp
    pval = np.mean(np.abs(perm) >= abs(stat))
    return float(stat), float(pval)


# ================================================================== #
# Doble bootstrap de la regresion truncada + cluster por caja
# ================================================================== #
def double_bootstrap(y, X, groups, rng, b1=B1, b2=B2, upper=UPPER):
    k = X.shape[1]
    beta_hat, sigma_hat, _ = fit_truncated(y, X, upper)

    # B1: correccion de sesgo (bootstrap parametrico de residuos truncados)
    mu_hat = X @ beta_hat
    b1_betas = np.full((b1, k), np.nan)
    for b in range(b1):
        ys = draw_truncated(mu_hat, sigma_hat, upper, rng)
        try:
            bb, _, _ = fit_truncated(ys, X, upper); b1_betas[b] = bb
        except Exception:
            pass
    beta_star_mean = np.nanmean(b1_betas, axis=0)
    beta_bc = 2 * beta_hat - beta_star_mean
    sigma_bc = sigma_hat  # sigma se mantiene (correccion centrada en beta)

    # B2: inferencia parametrica alrededor del estimador corregido
    mu_bc = X @ beta_bc
    b2_betas = np.full((b2, k), np.nan)
    for c in range(b2):
        ys = draw_truncated(mu_bc, sigma_bc, upper, rng)
        try:
            bb, _, _ = fit_truncated(ys, X, upper); b2_betas[c] = bb
        except Exception:
            pass
    ci_lo = np.nanpercentile(b2_betas, 2.5, axis=0)
    ci_hi = np.nanpercentile(b2_betas, 97.5, axis=0)
    se_boot = np.nanstd(b2_betas, axis=0)
    pval = np.array([2 * min(np.nanmean(b2_betas[:, j] <= 0),
                             np.nanmean(b2_betas[:, j] >= 0)) for j in range(k)])
    pval = np.clip(pval, 0, 1)

    # Bootstrap por CONGLOMERADO de caja (dependencia de panel, Du et al.)
    uc = np.unique(groups)
    clus_betas = np.full((b2, k), np.nan)
    for c in range(b2):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        try:
            bb, _, _ = fit_truncated(y[idx], X[idx], upper); clus_betas[c] = bb
        except Exception:
            pass
    ci_lo_cl = np.nanpercentile(clus_betas, 2.5, axis=0)
    ci_hi_cl = np.nanpercentile(clus_betas, 97.5, axis=0)
    se_cl = np.nanstd(clus_betas, axis=0)

    return {
        "beta_hat": beta_hat, "beta_bc": beta_bc, "sigma": sigma_hat,
        "se_boot": se_boot, "ci_lo": ci_lo, "ci_hi": ci_hi, "pval": pval,
        "se_cluster": se_cl, "ci_lo_cl": ci_lo_cl, "ci_hi_cl": ci_hi_cl,
    }


# ================================================================== #
# Utilidades de reporte de texto
# ================================================================== #
class Report:
    def __init__(self): self.buf = io.StringIO()
    def w(self, s=""): self.buf.write(str(s) + "\n")
    def hr(self, ch="="): self.w(ch * 78)
    def save(self, path):
        path.write_text(self.buf.getvalue(), encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    R = Report()
    R.hr(); R.w("CAPA 3 - INFERENCIA CONTEXTUAL (SEGUNDA ETAPA DEA)")
    R.w("Regresion truncada con doble bootstrap de panel (Simar-Wilson 2007 / Du et al. 2018)")
    R.w(f"Fuente: {DATA_PATH.name}  |  seed={SEED}  |  B1={B1}  B2={B2}")
    R.hr()

    df = pd.read_excel(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["ln_activo"] = np.log(df["total_activo_r"])
    y_all = df[DEP].values.astype(float)

    # -------------------- A. ANALISIS ESTADISTICO PREVIO -------------------- #
    R.w("\n[A] ANALISIS ESTADISTICO PREVIO")
    R.w(f"  Observaciones: {len(df)} caja-anio | dependiente: {DEP} en (0,1]")
    R.w(f"  Eficientes en el borde (score=1): {int((y_all>=0.999999).sum())} / {len(df)}")
    desc_vars = [DEP] + CONT + DUMMIES
    desc = df[desc_vars].agg(["mean", "std", "min", "median", "max"]).T
    desc["skew"] = df[desc_vars].skew()
    desc["kurtosis"] = df[desc_vars].kurtosis()
    R.w("\n  Descriptivos:")
    R.w(desc.round(4).to_string())
    corr = df[[DEP] + CONT].corr()
    R.w("\n  Correlaciones con la dependiente (score_corregido):")
    R.w(corr[DEP].drop(DEP).round(3).sort_values().to_string())

    # -------------------- B. SELECCION DE VARIABLES (VIF) ------------------- #
    R.w("\n[B] CRITERIO DE SELECCION DE VARIABLES")
    R.w("  Regla: (i) log del activo (asimetria/escala); (ii) VIF por pasos, umbral 5;")
    R.w("         (iii) dummies siempre incluidas; (iv) relevancia teorica.")
    vif_full = vif_series(df[CONT])
    R.w("\n  VIF del modelo COMPLETO (continuas):")
    R.w(vif_full.round(3).sort_values(ascending=False).to_string())
    keep, dropped, vif_final = stepwise_vif(df[CONT], VIF_THRESHOLD)
    R.w("\n  Variables retiradas por VIF (en orden):")
    for d, vv in dropped:
        R.w(f"    - {d}  (VIF={vv})")
    if not dropped:
        R.w("    (ninguna: todas por debajo del umbral)")
    R.w("\n  VIF del modelo FINAL:")
    R.w(vif_final.round(3).sort_values(ascending=False).to_string())
    final_cont = keep
    final_regs = final_cont + DUMMIES
    R.w(f"\n  >> Modelo final (Z): {final_regs}")

    # Matrices: se ESTANDARIZAN las continuas (media 0, sd 1) para condicionamiento
    # numerico y comparabilidad; las dummies quedan 0/1. Los coeficientes se
    # reportan en ambas escalas (los p-valores son invariantes a la escala).
    mu_c = df[final_cont].mean().values
    sd_c = df[final_cont].std().values
    Xc_std = ((df[final_cont] - df[final_cont].mean()) / df[final_cont].std()).values
    Xd = df[DUMMIES].values.astype(float)
    X_final = np.column_stack([np.ones(len(df)), Xc_std, Xd])
    names_final = ["const"] + final_cont + DUMMIES
    groups = df[ID].values
    nC = len(final_cont)

    def to_natural(vec):
        """Convierte coeficientes de escala estandarizada a unidades naturales."""
        out = np.array(vec, float).copy()
        slopes = np.array(vec[1:1 + nC], float)
        out[1:1 + nC] = slopes / sd_c
        out[0] = vec[0] - np.sum(slopes * mu_c / sd_c)
        return out

    # -------------------- C. TEST DE SEPARABILIDAD -------------------------- #
    R.w("\n[C] TEST DE SEPARABILIDAD (Daraio, Simar & Wilson, 2018) - version practica")
    Zc = df[final_cont].values.astype(float)
    sep_stat, sep_p = separability_test(df[DEP], Zc, rng, n_perm=2000)
    decision = ("Se RECHAZA la ausencia de asociacion Z-eficiencia (p<0.05): la 2a etapa "
                "es pertinente (condicion necesaria satisfecha)." if sep_p < 0.05 else
                "NO se rechaza (p>=0.05): Z no muestra asociacion significativa con la eficiencia.")
    R.w(f"  Estadistico = {sep_stat:.4f}   p-valor (permutacion) = {sep_p:.4f}")
    R.w(f"  Decision: {decision}")
    R.w("  Nota: version practica (asociacion con nulo por permutacion). El contraste")
    R.w("  frontera-vs-distribucion completo exige fronteras condicionales (Daraio-Simar 2005);")
    R.w("  si se rechazara la separabilidad estructural, se migraria a eficiencia condicional.")

    # -------------------- D. REGRESION TRUNCADA DOBLE BOOTSTRAP ------------- #
    R.w("\n[D] REGRESION TRUNCADA CON DOBLE BOOTSTRAP DE PANEL")
    R.w(f"  B1={B1} (correccion de sesgo de beta) | B2={B2} (inferencia) | truncacion superior en {UPPER}")
    res = double_bootstrap(y_all, X_final, groups, rng)
    coef_nat = to_natural(res["beta_bc"])
    tr = pd.DataFrame({
        "variable": names_final,
        "coef_std": res["beta_bc"],            # escala estandarizada (comparable)
        "coef_natural": coef_nat,              # unidades naturales (interpretacion)
        "se_bootstrap": res["se_boot"],
        "ic95_inf": res["ci_lo"], "ic95_sup": res["ci_hi"],
        "p_valor": res["pval"],
        "se_cluster_caja": res["se_cluster"],
        "ic95_inf_cluster": res["ci_lo_cl"], "ic95_sup_cluster": res["ci_hi_cl"],
    })
    tr["signif_5pct"] = tr["p_valor"] < 0.05
    R.w("  (continuas estandarizadas; coef_std comparable por 1 SD; coef_natural en unidades)")
    R.w("\n  Coeficientes (corregidos por sesgo) con IC 95% y p-valor bootstrap [escala estandarizada]:")
    R.w(tr[["variable", "coef_std", "coef_natural", "se_bootstrap", "ic95_inf", "ic95_sup",
            "p_valor", "signif_5pct"]].round(4).to_string(index=False))
    R.w("\n  IC cluster-robusto por caja (dependencia de panel) [escala estandarizada]:")
    R.w(tr[["variable", "coef_std", "se_cluster_caja",
            "ic95_inf_cluster", "ic95_sup_cluster"]].round(4).to_string(index=False))

    # -------------------- E. TOBIT DE REFERENCIA --------------------------- #
    R.w("\n[E] TOBIT (censura superior en 1) - REFERENCIA COMPARATIVA")
    tb_beta, tb_sigma, tb_se = fit_tobit(y_all, X_final)
    tb_t = tb_beta / tb_se
    tb_p = 2 * (1 - stats.norm.cdf(np.abs(tb_t)))
    tob = pd.DataFrame({"variable": names_final, "coef_std": tb_beta,
                        "coef_natural": to_natural(tb_beta), "se": tb_se,
                        "p_valor": tb_p, "signif_5pct": tb_p < 0.05})
    R.w(tob.round(4).to_string(index=False))
    R.w("\n  ADVERTENCIA: el Tobit se reporta solo como referencia. Sus errores estandar")
    R.w("  son invalidos aqui: los puntajes DEA son dependientes (frontera comun y ventanas),")
    R.w("  lo que viola el supuesto de independencia (Simar & Wilson, 2007). La inferencia")
    R.w("  valida es la del doble bootstrap del bloque [D].")

    # -------------------- F. DISCUSION ECONOMICA --------------------------- #
    R.w("\n[F] DISCUSION: signos esperados y significancia economica")
    disc_rows = []
    for _, r in tr.iterrows():
        v = r["variable"]
        exp = EXPECTED.get(v, None)
        obt = "+" if r["coef_natural"] > 0 else "-"
        coincide = "-" if exp is None else ("si" if exp == obt else "no")
        sig = "significativo" if r["signif_5pct"] else "no significativo"
        disc_rows.append({"variable": v, "signo_esperado": exp or "-",
                          "signo_obtenido": obt, "coincide": coincide,
                          "p_valor": round(float(r["p_valor"]), 4), "significancia": sig})
    disc = pd.DataFrame(disc_rows)
    R.w(disc.to_string(index=False))
    R.w("\n  Lectura de referencia (segun teoria):")
    lect = {
        "roa": "Rentabilidad del activo: mayor productividad de activos -> mayor eficiencia (+).",
        "roe": "Rentabilidad patrimonial: desempeno gerencial -> eficiencia (+).",
        "ratio_creditos_depositos": "Intensidad de intermediacion: mas colocacion por deposito -> eficiencia (+).",
        "apalancamiento_pasivo_capital_reservas": "Apalancamiento alto: mayor riesgo/menor holgura -> eficiencia (-).",
        "ln_activo": "Tamano: escala y experiencia -> eficiencia (+).",
        "dolarizacion_depositos": "Dolarizacion: exposicion cambiaria -> eficiencia (-).",
        "pbi_crec_real": "Ciclo: expansion del PBI eleva demanda de credito sano -> eficiencia (+).",
        "ipc_va": "Inflacion: entorno adverso, erosiona margenes -> eficiencia (-).",
        "d_pandemia": "Choque COVID 2020-2021 -> eficiencia (-).",
        "d_fintech": "Competencia digital post-2021 -> presion competitiva (-).",
    }
    for v in final_regs:
        if v in lect:
            R.w(f"    - {v}: {lect[v]}")

    R.w("\n" + "=" * 78)
    R.w("FIN DEL REPORTE")
    R.hr()

    # -------------------- EXPORTACION -------------------------------------- #
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        desc.round(6).to_excel(xw, sheet_name="analisis_previo")
        corr.round(4).to_excel(xw, sheet_name="correlaciones")
        pd.DataFrame({"variable": vif_full.index, "vif_completo": vif_full.values}
                     ).to_excel(xw, sheet_name="vif_seleccion", index=False)
        pd.DataFrame({"retirada": [d for d, _ in dropped],
                      "vif_al_retirar": [v for _, v in dropped]}
                     ).to_excel(xw, sheet_name="vif_retiradas", index=False)
        pd.DataFrame({"separabilidad": ["estadistico", "p_valor", "decision"],
                      "valor": [round(sep_stat, 4), round(sep_p, 4), decision]}
                     ).to_excel(xw, sheet_name="separabilidad", index=False)
        tr.round(6).to_excel(xw, sheet_name="truncada_doble_bootstrap", index=False)
        tob.round(6).to_excel(xw, sheet_name="tobit_referencia", index=False)
        disc.to_excel(xw, sheet_name="discusion", index=False)

    R.save(OUT_TXT)

    print(f"[05] Modelo final Z: {final_regs}")
    print(f"[05] Separabilidad: stat={sep_stat:.3f}, p={sep_p:.3f}")
    print("[05] Truncada (doble bootstrap) - significativos al 5%:")
    for _, r in tr.iterrows():
        if r["signif_5pct"]:
            print(f"     {r['variable']:<42s} coef_std={r['coef_std']:+.4f}  "
                  f"coef_nat={r['coef_natural']:+.4f}  p={r['p_valor']:.4f}")
    print(f"[05] Guardado: {OUT_XLSX}")
    print(f"[05] Reporte de texto: {OUT_TXT}")


if __name__ == "__main__":
    main()
