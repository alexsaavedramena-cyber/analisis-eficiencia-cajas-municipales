# Replicación — Tablas y figuras de la validación externa (ECR-SBS vs. DEA)

Este paquete reproduce, de forma **automática y a partir de un Excel guía**, las
tablas y figuras del reporte que contrasta las **clasificaciones de riesgo de
fortaleza financiera (SBS / Empresas Clasificadoras de Riesgo)** con el **ranking
de eficiencia DEA corregido por sesgo (KSW)** de las 12 CMAC.

## Contenido

| Archivo | Qué hace |
|---|---|
| `generar_guia_excel.py` | Crea el Excel guía uniendo scores KSW (de `outputs/tables/eficiencia_limpia.xlsx`) con las calificaciones SBS. Correr **una sola vez** o al actualizar datos. |
| `guia_scores_calificaciones.xlsx` | **Insumo único.** Hoja `datos` (scores + calificaciones), hoja `escala_notch` (leyenda) y `_LEEME`. |
| `replicar_tablas_figuras.py` | Lee el Excel guía y genera todas las tablas y figuras en `./salidas`. |
| `salidas/` | Resultados: `tabla1..4` (CSV), `tablas_generadas.xlsx`, `figura1_dispersion.png`, `figura2_brechas.png`. |

## Requisitos

```bash
pip install pandas numpy scipy matplotlib openpyxl
```

## Uso

```bash
# 1) (una vez) construir el Excel guía a partir de tus resultados
python generar_guia_excel.py

# 2) reproducir tablas y figuras
python replicar_tablas_figuras.py
```

Si más adelante cambian los scores o las calificaciones, tienes dos caminos:
- **editar a mano** `guia_scores_calificaciones.xlsx` (hoja `datos`) y volver a correr el paso 2; o
- volver a correr el paso 1 (si actualizaste `eficiencia_limpia.xlsx` o el diccionario de calificaciones dentro de `generar_guia_excel.py`).

## Proceso metodológico (qué calcula el script)

1. **Lectura** del Excel guía (hoja `datos`): una fila por CMAC con `score_ksw`
   y una columna por agencia (`apoyo, class, jcr, microrate, moodys, pcr`).
2. **Consolidación de la calificación**: cada letra se convierte a un valor
   ordinal `notch` (`A+=13 … E=1`); si varias agencias clasifican a la caja se
   **promedia** el notch; `RET` (retirada) se ignora; sin letras → *sin
   clasificación*.
3. **Ranking DEA**: puesto por `score_ksw` (1 = más eficiente).
4. **Estandarización (z)** de `score_ksw` y de `notch` sobre las cajas con
   calificación, y **brecha** = `z(eficiencia) − z(notch)`
   (positiva = eficiente pero peor calificada).
5. **Tabla 1** = leyenda escala→notch. **Tabla 2** = calificaciones vigentes con
   agencias y síntesis de nivel. **Tabla 3** = cruce DEA↔rating con la brecha.
   **Tabla 4** = correlaciones Pearson/Spearman/Kendall con p-valores (`scipy`).
6. **Figura 1** = dispersión notch (x) vs. eficiencia (y) con recta de tendencia
   (`np.polyfit`) y etiquetas coloreadas por la brecha; el eje X se re-etiqueta
   de número a letra. **Figura 2** = barras divergentes de la brecha por caja.

## Nota de fuentes

- Scores: `a-definitivo/outputs/tables/eficiencia_limpia.xlsx` → `ranking_corregido`.
- Calificaciones: portal SBS, *Resumen de clasificaciones de las ECR*
  (corte 2026-marzo, verificado con 2025-septiembre).
- La escala de fortaleza financiera sigue la Resolución SBS N.° 18400-2010.
