# Experimento — Ciclo profit–investment (rediseño 2026-06-03)

**Estado:** diseño limpio que supersede todo lo anterior (archivado en
`output/archive/2026-06-03_pre_redesign/` y `src/experiment/_archive/`).
**Motivación del rediseño:** los experimentos previos se enredaron por comparaciones con
*scope mismatch* (FN evaluado sobre todo el ciclo NBER vs baselines sobre la ventana de
crisis → FN parecía mejor de lo que era). Este rediseño usa **un solo harness en Python**
donde todos los modelos pronostican EXACTAMENTE los mismos objetivos.

## Pregunta y hallazgos (honestos)

Un oscilador no lineal del ciclo profit–investment, ¿describe y pronostica el ciclo
—incluidas las crisis— mejor que baselines lineales?

| Claim | Veredicto |
|---|---|
| **HAY un ciclo endógeno de sobreacumulación** (P7) | ✅ la CCF profits↔investment **oscila** (no decae): boom profits→investment (lag0 +0.67); giro: inversión pasada deprime profits (lag−4 −0.40). Robusto sin 2008/2020. **Hallazgo positivo central.** |
| El oscilador full-series produce un ciclo límite que describe la serie | ❌ **degenera** (colapsa a casi-punto fijo; RMSE≈desvío de los datos). No hay un único ciclo para 78 años. |
| Asimetría endógena (crash rápido/recuperación lenta) que el VAR no puede | ❌ skew(Δ)≈0 al excluir 2008/2020 (la asimetría eran los outliers) |
| **Hay predictibilidad de crisis a h corto sobre RW** | ✅ pero la captura el **VAR lineal** (U≈0.7–0.9 a h≥2, robusto en las 3 crisis) |
| El oscilador no lineal mejora al VAR de forma confiable | ❌ FN gana en 2008 (U 0.65–0.77) pero **es inestable** (explota en 2001); no generaliza |
| **2008 excede el rango de amplitud de todo el régimen previo** | ✅ **outlier** (ratio 1.03). 2020 fue grande pero no superó a 2008. |

**Síntesis:** la predictibilidad de crisis que existe es **lineal** (VAR > RW); la
maquinaria no lineal aporta inestabilidad, no skill confiable. El aporte del oscilador es
**descriptivo/estructural**: ubicar la frontera de amplitud (2008 redefinió la envolvente
de posguerra), no pronosticar.

**Encuadre (Astarita/Tapia):** "fuera de régimen de amplitud" NO es "shock exógeno" — es
la misma dinámica de acumulación en una fase descendente que el régimen ordinario (y el
modelo de amplitud fija) no alcanza.

## Diseño (3 piezas, deterministas, un solo harness)

- **`common.py`** — datos YoY, fechado NBER, osciladores FN/LV (scipy), z-score
  anti-leakage, ajuste single-shooting.
- **`p1_oscillation.py`** — oscilación característica: LV/FN por solver, trayectoria +
  retrato de fase. *(Descriptivo. El full-series degenera → pendiente: reajustar sobre
  una ventana representativa para mostrar un ciclo límite genuino.)*
- **`p2_amplitude_boundary.py`** — frontera de amplitud: 12 recesiones, ratio
  excursión/máx-previo, dentro vs outlier. **El hallazgo central.**
- **`models.jl`** (Julia) — capa de MODELOS: FN y LV ajustados por **solver single-shooting**
  y por **PINN inversa** (restricción blanda). Exporta `julia_forecasts.csv`.
- **`p3_forecast.py`** (Python) — evaluación del pronóstico: lee `julia_forecasts.csv`,
  agrega RW/media/VAR sobre los MISMOS targets (alineado), Theil's U + Diebold-Mariano.
  Arquitectura: modelos en Julia, evaluación en Python.

### Resultado P3 (con PINN inversa)
- **La PINN inversa regulariza el problema mal-condicionado y elimina los blow-ups del
  solver** (LV 2001: solver U=15–31 → PINN 0.90; FN 2001 recupera `b`=33 degenerado vs
  PINN `b`=0.5 sensato). FN_pinn es el oscilador más consistente (U<1 casi siempre).
- **PERO ninguna victoria U<1 es DM-significativa** (n≈8–12); los `*` son casi todos
  modelos *peores* que RW. "Le gana a RW" = sugestivo, no probado.
- VAR sigue siendo el lineal confiable (~0.8 a h≥2). Aporte de la PINN = método de
  inversión robusto, no skill predictivo mágico.

### Verificación del entrenamiento PINN (2026-06-03)
Verificado (loguear L_data vs L_física): **λ=1.0 es el punto justo** — λ=0.1 deja el residuo
físico dominante (L_phys/L_data≈5.7, params sin sentido); λ=10 aplasta el ajuste a datos y
distorsiona `c`. Convergencia: ambas losses se aplanan antes de 2500 épocas. Sin overfit:
params fast (2500/lr0.01/[16,16]) vs careful (10000/lr0.001) coinciden al 1–2% en los
coeficientes dinámicos. ⇒ settings rápidas confiables; **mantener λ=1.0**.
- **`p4_dynamical_eval.py`** — evaluación dinámica (toolkit testing-Goodwin, gradient
  matching/Ramsay-Hooker): período implícito vs observado, autovalores/tipo de ciclo,
  R² del campo de fase, robustez de parámetros entre ciclos.

## Resultado de la evaluación dinámica (P4)

El modelo también falla en su propia cancha dinámica (no solo en pronóstico):
- **Período implícito ~21–29 años vs observado ~3 años** → *el modo de falla de Harvie*
  (la frecuencia natural del oscilador no coincide con la del ciclo real).
- **R² del campo de fase ≈ 0 / negativo** (FN −0.6, LV 0.04): los datos no siguen el campo.
- **FN amortiguado, LV neutral** → régimen casi-lineal confirmado.
- **Parámetros inestables entre ciclos** (CV≈1): no hay parámetros estructurales estables.

⇒ Las dos tradiciones de evaluación (dinámica y pronóstico) coinciden: el oscilador no
gana su complejidad. Aporte = descriptivo (frontera de amplitud) + el protocolo de
evaluación mismo. *Caveat: el R² del campo depende del suavizado de du/dt (3 puntos, crudo;
splines à la Ramsay-Hooker lo refinarían); período y estabilidad son robustos a eso.*

### El modelo correcto: oscilador LINEAL (P4, modelo `LIN`)
Como profits e investment CO-MUEVEN con rezago (no antagonismo presa-depredador), se probó
un oscilador lineal 2D. Resultado positivo:
- **Mejor R² del campo** que LV/FN (0.10–0.11 vs 0.04 vs negativo) — describe mejor el movimiento.
- **Co-movement casi contemporáneo.** Medida ROBUSTA = correlación cruzada (Tapia): pico en
  lag 0 (corr 0.67), profits lideran **~1 trimestre** (lag+1 corr 0.65). Fase efectiva ~10°,
  cerca de co-movement (0°), NO los 90° presa-depredador que LV/FN imponen → por eso fallan.
  *(NOTA: la "fase 46° / lead 1.5a" del autovector del LIN era un artefacto — fase del modo
  lento × período ruidoso ~12a; descartada. La correlación cruzada manda.)*
- **Foco débilmente amortiguado** (Re=−0.03) = oscilación lineal manejada por ruido = un VAR.
⇒ Hallazgo positivo: el ciclo es una **oscilación lineal amortiguada, co-movement con profits
liderando ~1 trimestre (Tapia)**, no un ciclo presa-depredador. Por eso VAR gana y LV/FN fallan.
*Caveat: R² del campo aún bajo (~0.10, dinámica mayormente por ruido).*

## Notas metodológicas (para no volver a marearnos)

- **Ventana de evaluación importa:** sobre el *ciclo entero* (mayormente expansión) RW es
  imbatible; sobre el *episodio de crisis* (pico−6m → valle+12m) RW es débil y el VAR le
  gana. La pregunta "predecir la crisis" usa el episodio.
- **Siempre alinear targets** entre modelos antes de comparar (un solo harness).
- Osciladores: chequear estabilidad numérica (FN/LV explotan en algunas ventanas → U≫1).
