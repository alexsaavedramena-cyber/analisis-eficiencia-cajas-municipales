# -*- coding: utf-8 -*-
"""
replicar_tablas_figuras.py
--------------------------
Reproduce, a partir del Excel guía 'guia_scores_calificaciones.xlsx',
las tablas y figuras del reporte de contrastación externa
(clasificaciones ECR-SBS vs. ranking de eficiencia DEA de las CMAC).

Genera en la carpeta ./salidas :
    tabla1_escala_notch.csv
    tabla2_clasificaciones.csv
    tabla3_cruce_dea_rating.csv
    tabla4_correlaciones.csv
    figura1_dispersion.png
    figura2_brechas.png
    tablas_generadas.xlsx   (todas las tablas, una por hoja)

Dependencias:  pandas, numpy, scipy, matplotlib, openpyxl
    pip install pandas numpy scipy matplotlib openpyxl

Uso:
    1) (una vez) python generar_guia_excel.py      # crea el Excel guía
    2)           python replicar_tablas_figuras.py # reproduce tablas+figuras
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
AQUI    = os.path.dirname(os.path.abspath(__file__))
GUIA    = os.path.join(AQUI, "guia_scores_calificaciones.xlsx")   # entrada
SALIDAS = os.path.join(AQUI, "salidas")                            # salida
os.makedirs(SALIDAS, exist_ok=True)

# Paleta (idéntica a la del reporte)
NAVY = "#1F2A44"; GOLD = "#B8860B"; GREY = "#5B6472"; RED = "#B03A2E"; GREEN = "#1E6B3A"

# Escala de fortaleza financiera -> notch (número mayor = más sólida)
NOTCH = {"A+":13,"A":12,"A-":11,"B+":10,"B":9,"B-":8,
         "C+":7,"C":6,"C-":5,"D+":4,"D":3,"D-":2,"E":1}
AGENCIAS = ["apoyo","class","jcr","microrate","moodys","pcr"]
NOMBRE_AG = {"apoyo":"Apoyo & Asociados","class":"Class & Asociados","jcr":"JCR Latino América",
             "microrate":"MicroRate","moodys":"Moody’s Local PE","pcr":"PCR"}
UMBRAL_BRECHA = 0.9   # |brecha| > umbral se resalta en color

# ----------------------------------------------------------------------------
# PASO 0 — Leer el Excel guía
# ----------------------------------------------------------------------------
datos = pd.read_excel(GUIA, sheet_name="datos")
for a in AGENCIAS:
    if a not in datos.columns:
        datos[a] = ""
    datos[a] = datos[a].fillna("").astype(str).str.strip()

# ----------------------------------------------------------------------------
# PASO 1 — Consolidar calificación por caja: letras, agencias, notch promedio
# ----------------------------------------------------------------------------
def consolidar(row):
    letras, ags, notches = [], [], []
    for a in AGENCIAS:
        g = row[a]
        if g and g.upper() != "RET":
            letras.append(g); ags.append(f"{g} ({NOMBRE_AG[a]})")
            notches.append(NOTCH[g])
    if notches:
        # letras únicas conservando orden
        letras_u = list(dict.fromkeys(letras))
        return pd.Series({
            "grade": " / ".join(letras_u),
            "agencias": ", ".join(ags),
            "notch": float(np.mean(notches))})
    return pd.Series({"grade": "sin clasificación vigente", "agencias": "", "notch": np.nan})

datos = pd.concat([datos, datos.apply(consolidar, axis=1)], axis=1)

# Ranking DEA (1 = más eficiente)
datos["rank"] = datos["score_ksw"].rank(ascending=False, method="first").astype(int)

# ----------------------------------------------------------------------------
# PASO 2 — Estandarización (z) y brecha, solo sobre cajas con calificación
# ----------------------------------------------------------------------------
sub = datos.dropna(subset=["notch"]).copy()
sub["z_dea"] = (sub["score_ksw"] - sub["score_ksw"].mean()) / sub["score_ksw"].std(ddof=0)
sub["z_rat"] = (sub["notch"]     - sub["notch"].mean())     / sub["notch"].std(ddof=0)
sub["gap"]   = sub["z_dea"] - sub["z_rat"]     # + = eficiente pero peor calificada
gapmap = dict(zip(sub["cmac"], sub["gap"]))

# ----------------------------------------------------------------------------
# TABLA 1 — Escala -> notch (referencia)
# ----------------------------------------------------------------------------
tabla1 = pd.read_excel(GUIA, sheet_name="escala_notch")

# ----------------------------------------------------------------------------
# TABLA 2 — Clasificaciones vigentes
# ----------------------------------------------------------------------------
def nivel(g):
    if g.startswith("A"): return "Fortaleza alta"
    if g.startswith("B"): return "Fortaleza buena/media"
    if "sin" in g:        return "Sin clasificación"
    if g.startswith("C"): return "Fortaleza suficiente/limitada"
    return "Debilidad marcada"
tabla2 = (datos.sort_values("notch", ascending=False, na_position="last")
          [["cmac","grade","agencias","notch"]].copy())
tabla2["nivel_sintesis"] = datos.sort_values("notch", ascending=False, na_position="last")["grade"].map(nivel)
tabla2 = tabla2.rename(columns={"cmac":"CMAC","grade":"Clasificacion","agencias":"Agencias","notch":"Notch"})

# ----------------------------------------------------------------------------
# TABLA 3 — Cruce DEA <-> rating
# ----------------------------------------------------------------------------
tabla3 = datos.sort_values("rank")[["rank","cmac","score_ksw","grade","notch"]].copy()
tabla3["brecha"] = tabla3["cmac"].map(gapmap)
tabla3 = tabla3.rename(columns={"rank":"Puesto_DEA","cmac":"CMAC","score_ksw":"Eficiencia_KSW",
                                "grade":"Clasificacion","notch":"Notch","brecha":"Brecha_z"})
tabla3["Eficiencia_KSW"] = tabla3["Eficiencia_KSW"].round(3)
tabla3["Notch"] = tabla3["Notch"].round(1)
tabla3["Brecha_z"] = tabla3["Brecha_z"].round(2)

# ----------------------------------------------------------------------------
# TABLA 4 — Correlaciones (eficiencia vs notch)
# ----------------------------------------------------------------------------
rho,  p_rho  = stats.spearmanr(sub["score_ksw"], sub["notch"])
tau,  p_tau  = stats.kendalltau(sub["score_ksw"], sub["notch"])
pear, p_pear = stats.pearsonr(sub["score_ksw"], sub["notch"])
tabla4 = pd.DataFrame({
    "Coeficiente": ["Pearson r","Spearman rho","Kendall tau"],
    "Valor":  [round(pear,3), round(rho,3), round(tau,3)],
    "p_valor":[round(p_pear,3), round(p_rho,3), round(p_tau,3)],
    "Que_capta":["Asociación lineal entre valores",
                 "Coincidencia de los órdenes (ranking)",
                 "Concordancia de pares (robusta con n pequeño)"]})

# ----------------------------------------------------------------------------
# FIGURA 1 — Dispersión eficiencia vs clasificación
# ----------------------------------------------------------------------------
def color_por_brecha(g):
    return RED if g > UMBRAL_BRECHA else (GREEN if g < -UMBRAL_BRECHA else GREY)

fig, ax = plt.subplots(figsize=(7.4,5.0), dpi=170)
x = sub["notch"].values; y = sub["score_ksw"].values
ax.scatter(x, y, s=75, color=NAVY, zorder=3, edgecolor="white", linewidth=0.8)
b, a = np.polyfit(x, y, 1)                       # recta de tendencia
xs = np.linspace(x.min()-0.4, x.max()+0.4, 50)
ax.plot(xs, a + b*xs, color=GOLD, lw=2, zorder=2, label=f"Ajuste lineal (r = {pear:.2f})")
for _, r in sub.iterrows():
    lab = r["cmac"].replace("CMAC ", "")
    ax.annotate(lab, (r["notch"], r["score_ksw"]), xytext=(r["notch"]+0.12, r["score_ksw"]),
                fontsize=8.2, color=color_por_brecha(r["gap"]), va="center")
ax.set_xlabel("Clasificación de fortaleza financiera (notch; mayor = más sólida)", fontsize=9.5)
ax.set_ylabel("Eficiencia DEA corregida por sesgo (KSW)", fontsize=9.5)
# Re-etiquetar el eje X de número a letra (según lo presente en los datos)
ticks = sorted(sub["notch"].unique())
inv = {v:k for k,v in NOTCH.items()}
ax.set_xticks(ticks)
ax.set_xticklabels([inv.get(t, f"{t:.1f}") if float(t).is_integer() else f"{t:.1f}" for t in ticks], fontsize=8.5)
ax.grid(True, ls=":", alpha=.5); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("Figura 1. Eficiencia técnica (DEA) frente a clasificación de riesgo ECR-SBS",
             fontsize=10, color=NAVY, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(SALIDAS, "figura1_dispersion.png"), bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# FIGURA 2 — Barras divergentes de la brecha
# ----------------------------------------------------------------------------
g = sub.sort_values("gap")
fig, ax = plt.subplots(figsize=(7.4,5.0), dpi=170)
cols = [color_por_brecha(v) for v in g["gap"]]
labs = [c.replace("CMAC ", "") for c in g["cmac"]]
ax.barh(labs, g["gap"], color=cols, edgecolor="white")
ax.axvline(0, color=NAVY, lw=1)
for i, v in enumerate(g["gap"]):
    ax.text(v + (0.05 if v>=0 else -0.05), i, f"{v:+.2f}",
            va="center", ha="left" if v>=0 else "right", fontsize=8, color="#333")
ax.set_xlabel("Brecha = z(eficiencia) − z(clasificación)   → derecha: eficiente pero peor calificada", fontsize=8.8)
ax.set_xlim(g["gap"].min()-0.5, g["gap"].max()+0.6)
ax.set_title("Figura 2. Brecha estandarizada entre eficiencia y clasificación por caja",
             fontsize=10, color=NAVY, weight="bold")
ax.grid(True, axis="x", ls=":", alpha=.5)
plt.tight_layout()
fig.savefig(os.path.join(SALIDAS, "figura2_brechas.png"), bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# GUARDAR TABLAS (CSV + un Excel con una hoja por tabla)
# ----------------------------------------------------------------------------
tabla1.to_csv(os.path.join(SALIDAS,"tabla1_escala_notch.csv"), index=False, encoding="utf-8-sig")
tabla2.to_csv(os.path.join(SALIDAS,"tabla2_clasificaciones.csv"), index=False, encoding="utf-8-sig")
tabla3.to_csv(os.path.join(SALIDAS,"tabla3_cruce_dea_rating.csv"), index=False, encoding="utf-8-sig")
tabla4.to_csv(os.path.join(SALIDAS,"tabla4_correlaciones.csv"), index=False, encoding="utf-8-sig")
with pd.ExcelWriter(os.path.join(SALIDAS,"tablas_generadas.xlsx"), engine="openpyxl") as xw:
    tabla1.to_excel(xw, sheet_name="T1_escala_notch", index=False)
    tabla2.to_excel(xw, sheet_name="T2_clasificaciones", index=False)
    tabla3.to_excel(xw, sheet_name="T3_cruce", index=False)
    tabla4.to_excel(xw, sheet_name="T4_correlaciones", index=False)

# ----------------------------------------------------------------------------
# RESUMEN EN CONSOLA
# ----------------------------------------------------------------------------
print("="*70)
print("REPLICACIÓN COMPLETA. Archivos en:", SALIDAS)
print("="*70)
print("\nTABLA 3 — Cruce DEA <-> rating:")
print(tabla3.to_string(index=False))
print(f"\nTABLA 4 — Correlaciones (n={len(sub)} cajas con clasificación):")
print(tabla4.to_string(index=False))
print("\nCajas sin clasificación vigente:",
      ", ".join(datos.loc[datos['notch'].isna(),'cmac']) or "ninguna")
