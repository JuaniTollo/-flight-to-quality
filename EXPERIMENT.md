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

---

## Corrección metodológica (2026-06-06): transformación YoY vs QoQ

Los resultados de CCF y estudio de evento de arriba usan **YoY = `pct_change(4)`**. Como
YoY es una diferencia a 4 trimestres, inyecta estructura de orden 4 e **infla el feedback
en lag−4 ~2×** — justo donde cae la señal. Recomputado sobre **QoQ = `pct_change(1)`**
(diferencia a 1 trimestre, sin esa estructura) el patrón **sobrevive pero a magnitud
honesta**:

| | lag0 (co-movimiento) | lag−4 cerca de crisis | contraste crisis−expansión (lag−4) |
|---|---|---|---|
| YoY | +0.72 | −0.63 | −0.53 |
| **QoQ (honesto)** | +0.62 | **−0.29** | **−0.29** |

⇒ **Usar QoQ como transform principal; las magnitudes históricas YoY van con un haircut
~2×.** El contraste crisis-vs-expansión es cualitativamente robusto (más fuerte en crisis),
pero menor de lo que sugería el YoY.

## Pieza 7 — CCF de sobreacumulación (`p7_overaccumulation.py`)

Estadística descriptiva pura (sin modelo): la correlación cruzada profits↔investment
**oscila** (cruza a negativo), no decae → hay ciclo endógeno. Co-movimiento contemporáneo
**dominante** (lag0) y feedback negativo retardado (lag−4) modesto. Fase efectiva ~10°
(co-movimiento), no los 90° presa-depredador. Documento metodológico: `p7_doc.py` (genera
`output/experiment/p7_metodologia_ccf.docx`).

## Pieza 8 — Estudio de evento de crisis (`p8_event_study.py`)

Superposed-epoch analysis sobre las recesiones NBER (11 usables, ±6 trim del fondo de
inversión):
- **Buildup de sobreacumulación pre-crisis:** la brecha inversión−ganancia es significativa
  (bootstrap IC95) en τ=−6,−5,−4 (1–1.5 años antes del fondo). Coincide con Tapia Granados
  (2012): la ganancia cae 4–5 trimestres antes de la recesión → **re-confirmación con método
  independiente**, no descubrimiento.
- **Feedback presa-depredador concentrado en crisis:** CCF condicional lag−4 cerca de crisis
  vs expansión = −0.29 vs 0.00 (QoQ; −0.63 vs −0.10 en YoY inflado). Emparejamiento
  within-window (corrige un bug de cruce de bordes que subestimaba el efecto).
- **"Profits lidera la caída": NO robusto** — el lead varía 0–2 trimestres según el ancla.
- Caveats: n=11, σ grandes; el lado post-fondo del buildup está contaminado por efecto base.

## Pieza 9 — Oscilador como régimen de crisis (`p9_*.py`)

Hipótesis: si el feedback presa-depredador vive cerca de las crisis, LV/FN deberían
describir la dinámica mejor ahí que en expansión.
- **`p9_crisis_regime` + `p9_ablate` (288 celdas):** R² del campo de fase mayor en crisis
  que en expansión para los osciladores NO lineales (FN ΔR²≈+0.23 a +0.32; LV +0.07 a +0.14),
  **ausente para el lineal LIN** (Δ≈−0.03) → el contraste es específico de la no-linealidad.
  Robusto a ancla (salvo ancla=pico) y ancho de ventana. **Pero no alcanza significancia**
  (n=8 pares, p≈0.73), el RMSE de trayectoria no separa regímenes, y depende fuerte de 2008.
- **`p9_placebo` (falsificación):** el buen ajuste de LV/FN a las crisis es **trivial** — un
  oscilador de 2–4 parámetros ajusta cualquier ventana de 13 trim comparablemente; el margen
  aparente es un **confound de amplitud** (las crisis tienen ~2.6× amplitud → mejor S/R). En
  el R² de campo no hay edge vs placebo. ⇒ "describe la crisis mejor" es, en bondad de ajuste,
  vacío.
- **`p9_pooled` (estructura compartida):** un único θ presa-depredador describe las 11 crisis
  perdiendo +37% de ajuste (LV) / +74% (FN) vs el ajuste por-crisis → dinámica de crisis
  compartida plausible (no idiosincrática). Caveat: falta el placebo del pooling.

## Pieza 10 — Kernel de lag distribuido / delay estocástico (`p10_*.py`)

Reencuadre: el feedback no es un retardo de τ fijo ni un oscilador de fase fija, sino un
**kernel de lag distribuido**. Ver `notes/P10_CONSOLIDADO.md`. Veredicto en tres claims:
- **Lag central μ≈4 trim (1 año): identificable y robusto** (convergencia de 5 métodos).
- **Forma distribuida: real pero no le gana al delay puntual** (ΔAICc prefiere τ=4; ancho `s`
  con perfil de verosimilitud plano, CI95(s)=[0,4]).
- **Delay estocástico (dispersión dentro de la crisis) y fluctuación entre crisis: NO
  identificables** (SD-entre-crisis del mecanismo β_lag4 = 0.001; el muro es el SNR: el
  feedback explica ~3.2% de la varianza vs ~35% el co-movimiento).
- Ranking: **VAR(p≥4-5)** reproduce la joroba 4-5 sin supervisión (es un kernel de lag libre);
  **LV/FN/VAR(1) de fase fija quedan refutados** (su CCF es un coseno rígido, no pulso+joroba).
- Distinción clave: los datos hablan de dispersión **entre** crisis de un lag por-crisis
  (media 4.2, sd 0.9), **no** de dispersión **dentro** de cada crisis. Separarlas requiere
  varias respuestas de impulso por episodio (límite duro).

## Pieza 11 — Modelo físico: cadena de maduración (`p11_physical_delay.py`)

Modelo físico autónomo y diferenciable que encarna el delay distribuido: la inversión
deprime la ganancia tras **madurar por etapas** (cascada de ODEs `dmⱼ/dt=(mⱼ₋₁−mⱼ)/θ`), lo
que por el *linear chain trick* equivale a un kernel Gamma (media n·θ, dispersión √n·θ).
Calibrado por su solución analítica (filtro) sobre las crisis (QoQ):
- **El centro del lag de maduración se identifica en ~1 año** (R² del ajuste pica en μ≈4–5 trim).
- **El ancho de la distribución NO se identifica** (R² varía 0.013 entre kernel ancho y casi
  puntual) → confirma, desde un modelo físico, la frontera de identificabilidad de P10.

## Síntesis general (2026-06-07)

1. Existe un ciclo de sobreacumulación con un feedback retardado **centrado en ~1 año**,
   más intenso en las vecindades de crisis (robusto), pero **débil** frente al co-movimiento
   contemporáneo y de magnitud menor de lo que sugería el YoY.
2. La **forma** del feedback (pulso contemporáneo + masa negativa a ~1 año) **refuta los
   osciladores de fase fija** (LV/FN/VAR1) y es capturada por un kernel de lag distribuido
   (≡ VAR(p≥4) / modelo físico de maduración).
3. La parte **estocástica/distribucional** de la hipótesis (ancho del delay, fluctuación
   entre crisis) es **estructuralmente no identificable** con esta data (n≈9 crisis, SNR≈3%).
   El resultado robusto es **dónde está esa frontera de identificabilidad**, caracterizada
   con un modelo físico diferenciable.
4. Consistente con todo lo previo: el aporte es **descriptivo/estructural + metodológico**,
   no de pronóstico (el VAR lineal sigue siendo el baseline a batir, y no se lo bate).
