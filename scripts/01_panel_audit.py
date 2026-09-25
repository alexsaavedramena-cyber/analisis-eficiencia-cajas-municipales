#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Fase 1 - Preparacion y auditoria del panel
Guia Metodologica Definitiva (v4) - Analisis DEA de eficiencia en CMAC (2002-2025)
--------------------------------------------------------------------------------
Modulo: 01_panel_audit.py

Objetivo (Fase 1 de la guia):
    Entregar a la frontera un panel limpio, con roles de variable fijados y la
    restriccion dimensional verificada.

Verificaciones que ejecuta:
    (i)   Estructura del panel: 286 observaciones caja-anio, 12 cajas 2002-2023
          y 11 en 2024-2025 (salida de CMAC Sullana tras 2023).
    (ii)  Ausencia de nulos, ceros estructurales y valores negativos en las seis
          variables del nucleo.
    (iii) Isotonia insumo-producto via correlaciones de Pearson y Spearman
          (incluida la fila del bad output cartera_atrasada_r).
    (iv)  Regla dimensional de Cooper (m + s <= N/3) anio por anio para el nucleo
          (m+s=6) y los contrastes reducidos D (m+s=2), C (m+s=3), A (m+s=4).

Salida:
    outputs/tables/auditoria_panel.xlsx
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Rutas
# ------------------------------------------------------------------ #
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "Libro1.xlsx"
OUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUT_PATH = OUT_DIR / "auditoria_panel.xlsx"

# ------------------------------------------------------------------ #
# Especificacion de variables (verificada contra la base, guia v4)
# ------------------------------------------------------------------ #
ID_COLS = ["cmac", "anio"]

INPUTS = ["g_personal_r", "n_oficinas"]                       # Etapa I (insumos)
INTERMEDIATE = "total_depositos_r"                            # enlace de red
GOOD_OUTPUTS = ["creditos_vigentes_r", "ingresos_finan_r"]   # Etapa II (deseados)
BAD_OUTPUT = "cartera_atrasada_r"                             # Etapa II (no deseado)

CORE_VARS = INPUTS + [INTERMEDIATE] + GOOD_OUTPUTS + [BAD_OUTPUT]  # 6 variables

# Contrastes parsimoniosos (m+s): dimensionalidad de la frontera
#   Nucleo: 2 insumos + 1 intermedio (dep) + 2 buenos + 1 malo -> para Cooper
#           usamos insumos del sistema (2) + productos del sistema (3) = 5,
#           mas el enlace tratado como variable => la guia declara m+s = 6.
#   D (m+s=2): 1 insumo -> 1 producto.
#   C (m+s=3): 2 insumos -> 1 producto (o 1->2).
#   A (m+s=4): 2 insumos -> 2 productos.
COOPER_SPECS = {
    "Nucleo (m+s=6)": 6,
    "Contraste A (m+s=4)": 4,
    "Contraste C (m+s=3)": 3,
    "Contraste D (m+s=2)": 2,
}


def load_panel() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=0)
    df = df.sort_values(ID_COLS).reset_index(drop=True)
    return df


def audit_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Composicion anual del panel y confirmacion de la salida de Sullana."""
    comp = (
        df.groupby("anio")
        .agg(n_cajas=("cmac", "nunique"), n_obs=("cmac", "size"))
        .reset_index()
    )
    cajas_por_anio = df.groupby("anio")["cmac"].apply(lambda s: set(s))
    all_cajas = set(df["cmac"].unique())
    comp["cajas_ausentes"] = comp["anio"].map(
        lambda a: ", ".join(sorted(all_cajas - cajas_por_anio[a])) or "(ninguna)"
    )
    return comp


def audit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Nulos, ceros estructurales, negativos y descriptivos por variable."""
    rows = []
    for v in CORE_VARS:
        col = df[v]
        rows.append(
            {
                "variable": v,
                "rol": _role(v),
                "n_nulos": int(col.isna().sum()),
                "n_ceros": int((col == 0).sum()),
                "n_negativos": int((col < 0).sum()),
                "minimo": float(col.min()),
                "media": float(col.mean()),
                "mediana": float(col.median()),
                "maximo": float(col.max()),
                "cv": float(col.std() / col.mean()) if col.mean() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["OK_sin_nulos_ceros_neg"] = (
        (out["n_nulos"] == 0) & (out["n_ceros"] == 0) & (out["n_negativos"] == 0)
    )
    return out


def _role(v: str) -> str:
    if v in INPUTS:
        return "insumo (Etapa I)"
    if v == INTERMEDIATE:
        return "intermedio / enlace"
    if v in GOOD_OUTPUTS:
        return "producto deseado (Etapa II)"
    if v == BAD_OUTPUT:
        return "producto NO deseado (Etapa II)"
    return "?"


def correlations(df: pd.DataFrame):
    """Matrices de correlacion Pearson y Spearman sobre el nucleo."""
    sub = df[CORE_VARS]
    pear = sub.corr(method="pearson")
    spear = sub.corr(method="spearman")
    return pear, spear


def isotonia_reading(pear: pd.DataFrame) -> pd.DataFrame:
    """
    Lectura de isotonia: correlacion de cada insumo con cada producto y del bad
    output con el resto (debe ser positiva y fuerte bajo disponibilidad debil).
    """
    good = INPUTS + [INTERMEDIATE] + GOOD_OUTPUTS
    rows = []
    for a in good:
        for b in good:
            if a >= b:
                continue
            r = pear.loc[a, b]
            rows.append(
                {
                    "par": f"{a} ~ {b}",
                    "tipo": "insumo/producto bueno",
                    "pearson": round(float(r), 4),
                    "isotonia_ok": bool(r > 0.5),
                }
            )
    # Fila del bad output frente a todas las variables buenas
    for a in good:
        r = pear.loc[BAD_OUTPUT, a]
        rows.append(
            {
                "par": f"{BAD_OUTPUT} ~ {a}",
                "tipo": "bad output vs bueno (disp. debil)",
                "pearson": round(float(r), 4),
                "isotonia_ok": bool(r > 0.5),
            }
        )
    return pd.DataFrame(rows)


def cooper_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regla de Cooper: m + s <= N / 3, anio por anio, en modo contemporaneo
    (N = numero de DMU efectivas en el anio).
    """
    rows = []
    for anio, g in df.groupby("anio"):
        N = g["cmac"].nunique()
        umbral = N / 3.0
        row = {"anio": int(anio), "N_DMU": int(N), "umbral_N/3": round(umbral, 3)}
        for name, ms in COOPER_SPECS.items():
            row[name] = "cumple" if ms <= umbral else "NO cumple"
        rows.append(row)
    return pd.DataFrame(rows)


def summary_flags(struct, qual, cooper) -> pd.DataFrame:
    """Tabla resumen de banderas de aceptacion de la Fase 1."""
    n_obs = int(struct["n_obs"].sum())
    ok_286 = n_obs == 286
    ok_12_2023 = bool(
        (struct.loc[struct.anio.between(2002, 2023), "n_cajas"] == 12).all()
    )
    ok_11_2425 = bool(
        (struct.loc[struct.anio.between(2024, 2025), "n_cajas"] == 11).all()
    )
    ok_quality = bool(qual["OK_sin_nulos_ceros_neg"].all())
    ok_nucleo_all = bool((cooper["Nucleo (m+s=6)"] == "cumple").all())
    ok_nucleo_2425 = bool(
        (cooper.loc[cooper.anio.between(2024, 2025), "Nucleo (m+s=6)"] == "cumple").all()
    )
    rows = [
        ("Total de observaciones = 286", ok_286, f"n_obs={n_obs}"),
        ("12 cajas en 2002-2023", ok_12_2023, ""),
        ("11 cajas en 2024-2025 (salida Sullana)", ok_11_2425, ""),
        ("Sin nulos / ceros / negativos en las 6 variables", ok_quality, ""),
        ("Nucleo (m+s=6) cumple Cooper en TODOS los anios (contemporaneo)",
         ok_nucleo_all,
         "en 2024-2025, con N=11, el nucleo contemporaneo no lo cumple: por eso "
         "los niveles se estiman con ventanas w=3 (~34-36 pseudo-unidades)"),
        ("Nucleo (m+s=6) cumple Cooper en 2024-2025 contemporaneo",
         ok_nucleo_2425, ""),
    ]
    return pd.DataFrame(rows, columns=["verificacion", "resultado", "nota"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[01] Cargando panel: {DATA_PATH}")
    df = load_panel()
    print(f"[01] Observaciones: {len(df)} | cajas: {df['cmac'].nunique()} | "
          f"anios: {df['anio'].min()}-{df['anio'].max()}")

    struct = audit_structure(df)
    qual = audit_quality(df)
    pear, spear = correlations(df)
    iso = isotonia_reading(pear)
    cooper = cooper_rule(df)
    resumen = summary_flags(struct, qual, cooper)

    # Consola: banderas principales
    print("\n[01] Banderas de aceptacion (Fase 1):")
    for _, r in resumen.iterrows():
        flag = "OK " if r["resultado"] else "!! "
        print(f"    {flag} {r['verificacion']}")

    print("\n[01] Rango de correlaciones de Pearson (variables buenas):")
    good = INPUTS + [INTERMEDIATE] + GOOD_OUTPUTS
    block = pear.loc[good, good]
    off = block.where(~np.eye(len(good), dtype=bool))
    print(f"    min={np.nanmin(off.values):.3f}  max={np.nanmax(off.values):.3f}")
    print("[01] Correlacion Pearson del bad output con las variables buenas:")
    print(f"    min={pear.loc[BAD_OUTPUT, good].min():.3f}  "
          f"max={pear.loc[BAD_OUTPUT, good].max():.3f}")

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        resumen.to_excel(xw, sheet_name="resumen_aceptacion", index=False)
        struct.to_excel(xw, sheet_name="composicion_anual", index=False)
        qual.to_excel(xw, sheet_name="calidad_variables", index=False)
        pear.round(4).to_excel(xw, sheet_name="corr_pearson")
        spear.round(4).to_excel(xw, sheet_name="corr_spearman")
        iso.to_excel(xw, sheet_name="isotonia", index=False)
        cooper.to_excel(xw, sheet_name="regla_cooper", index=False)

    print(f"\n[01] Reporte de auditoria guardado en: {OUT_PATH}")


if __name__ == "__main__":
    main()
