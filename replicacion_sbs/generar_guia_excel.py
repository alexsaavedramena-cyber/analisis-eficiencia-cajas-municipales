# -*- coding: utf-8 -*-
"""
generar_guia_excel.py
---------------------
Construye el Excel guía 'guia_scores_calificaciones.xlsx' que alimenta al
script de replicación. Une:
  (a) los scores DEA corregidos por sesgo (KSW), tomados de
      outputs/tables/eficiencia_limpia.xlsx  -> hoja 'ranking_corregido'
  (b) las calificaciones de riesgo de fortaleza financiera de las CMAC,
      extraídas del portal SBS (corte 2026-marzo, verificado con 2025-sep).

Solo hay que correrlo una vez (o cada vez que actualices scores/calificaciones).
"""
import os
import pandas as pd

AQUI   = os.path.dirname(os.path.abspath(__file__))
RANKING = r"D:/Datos SBS/datos_dea_1/a-definitivo/outputs/tables/eficiencia_limpia.xlsx"
SALIDA  = os.path.join(AQUI, "guia_scores_calificaciones.xlsx")

# --- 1) Scores KSW desde el archivo oficial de resultados -------------------
rk = pd.read_excel(RANKING, sheet_name="ranking_corregido")[["cmac", "score_corregido_prom"]]
# Nombre corto homogéneo para figuras/tablas
rk["cmac"] = rk["cmac"].replace({"Caja Municipal de Crédito Popular Lima": "CMCP Lima"})

# --- 2) Calificaciones SBS por agencia (celda vacía = no la clasifica) ------
# Agencias: apoyo, class, jcr, microrate, moodys, pcr. 'RET' = retirada.
calif = {
 "CMAC Arequipa":  dict(jcr="A-", pcr="A-"),
 "CMAC Cusco":     dict(apoyo="A-", jcr="A-", pcr="A-", moodys="RET"),
 "CMAC Huancayo":  dict(jcr="A-", moodys="B+", pcr="A-"),
 "CMAC Trujillo":  dict(apoyo="B+", moodys="B+"),
 "CMAC Ica":       dict(jcr="B+", pcr="B+"),
 "CMAC Piura":     dict(microrate="B", moodys="B"),
 "CMAC Maynas":    dict(apoyo="B-", moodys="B-"),
 "CMAC Tacna":     dict(jcr="C", moodys="C"),
 "CMAC Paita":     dict(apoyo="C", microrate="C"),
 "CMCP Lima":      dict(jcr="C+", microrate="C+"),
 "CMAC Del Santa": dict(apoyo="C-", microrate="D+"),
 "CMAC Sullana":   dict(),   # sin clasificación vigente
}
AGENCIAS = ["apoyo", "class", "jcr", "microrate", "moodys", "pcr"]

filas = []
for _, r in rk.iterrows():
    d = {"cmac": r["cmac"], "score_ksw": round(float(r["score_corregido_prom"]), 6)}
    g = calif.get(r["cmac"], {})
    for a in AGENCIAS:
        d[a] = g.get(a, "")
    filas.append(d)
datos = pd.DataFrame(filas, columns=["cmac", "score_ksw"] + AGENCIAS)

# --- 3) Hoja de referencia de la escala -> notch ----------------------------
escala = pd.DataFrame({
 "categoria": ["A+","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","E"],
 "notch":     [13,12,11,10,9,8,7,6,5,4,3,2,1],
 "significado":[
   "Fortaleza muy alta","Fortaleza muy alta","Fortaleza muy alta",
   "Buena fortaleza","Buena fortaleza","Buena fortaleza",
   "Fortaleza suficiente/limitada","Fortaleza suficiente/limitada","Fortaleza suficiente/limitada",
   "Debilidades importantes","Debilidades importantes","Debilidades importantes",
   "Insolvencia / riesgo severo"]})

lee = pd.DataFrame({"guia":[
 "Hoja 'datos': una fila por CMAC.",
 "score_ksw = eficiencia DEA corregida por sesgo (Kneip-Simar-Wilson), 0 a 1.",
 "Columnas apoyo..pcr = calificacion de fortaleza financiera por agencia (vacio = no clasifica).",
 "'RET' = clasificacion retirada; el script la ignora al promediar.",
 "Para actualizar: reemplaza score_ksw y/o las letras, guarda, y corre replicar_tablas_figuras.py.",
]})

with pd.ExcelWriter(SALIDA, engine="openpyxl") as xw:
    datos.to_excel(xw, sheet_name="datos", index=False)
    escala.to_excel(xw, sheet_name="escala_notch", index=False)
    lee.to_excel(xw, sheet_name="_LEEME", index=False)

print("Guia generada:", SALIDA)
print(datos.to_string(index=False))
