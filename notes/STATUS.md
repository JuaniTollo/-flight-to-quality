# Estado del proyecto y cómo seguir

Documento de orientación para recuperar el contexto. El detalle cuantitativo está en
[`EXPERIMENT.md`](../EXPERIMENT.md) y [`notes/P10_CONSOLIDADO.md`](P10_CONSOLIDADO.md);
acá va el mapa: qué se probó, qué quedó en pie, qué descartar, y los próximos pasos.

## 1. Pregunta y resultado afirmativo (a hoy)

El ciclo económico ganancias–inversión (profit–investment), en la tradición de
sobreacumulación (Astarita/Tapia): la inversión, al madurar, deprime la tasa de ganancia
**con retardo**. La afirmación que sostienen los datos:

> El feedback de sobreacumulación inversión→ganancia es un **retardo de maduración
> distribuido centrado en ~1 año**, más intenso cerca de las crisis. Lo captura un
> **modelo físico de cadena de maduración** (delay distribuido, ODE diferenciable);
> los osciladores de fase fija (Lotka-Volterra / FitzHugh-Nagumo) **no** pueden generarlo.

Números honestos (transform **QoQ**, que es el insesgado; ver §4): co-movimiento
contemporáneo lag0 ≈ +0.6 (dominante, ~35% var); feedback de maduración centrado en
~4 trimestres (β ≈ −0.25 a −0.3); contraste crisis-vs-expansión presente pero modesto.

## 2. Mapa de piezas (qué hace cada una)

| Pieza | Script | Qué muestra | Estado |
|---|---|---|---|
| P1 | `oscillation.jl` | oscilación característica LV/FN (solver) | full-series degenera |
| P2 | `p2_amplitude_boundary.py` | frontera de amplitud (2008 outlier) | ✅ descriptivo |
| P3 | `models.jl` + `p3_forecast.py` | pronóstico FN/LV (solver + PINN inversa) vs RW/VAR | sin ventaja sobre VAR |
| P4 | `p4_dynamical_eval.py` | evaluación dinámica (gradient matching, R² campo) | osciladores no describen el campo |
| P5/P6 | `p5_nonstationary.py`, `p6_tvp_forecast.py` | no-estacionariedad, VAR-TVP | contexto |
| P7 | `p7_overaccumulation.py` (+ `p7_doc.py`) | CCF profits↔investment (oscila, no decae) | ✅ patrón empírico |
| P8 | `p8_event_study.py` | estudio de evento de crisis; buildup pre-crisis | ✅ (= Tapia 2012) |
| P9 | `p9_crisis_regime.py`, `p9_ablate.py`, `p9_placebo.py`, `p9_pooled.py` | ¿oscilador = régimen de crisis? | ⚠️ ver §3 |
| P10 | `p10_*.py` (+ `notes/P10_CONSOLIDADO.md`) | bench del kernel de lag distribuido / delay estocástico | ✅ identificabilidad |
| P11 | `p11_physical_delay.py`, `p11_pinn_maturation.jl` | modelo físico de maduración; calibración por PINN inversa | 🔜 en progreso (ver §5) |

## 3. Qué quedó en pie vs qué descartar (para no re-probar)

**Afirmativo (usar):**
- CCF oscila → hay ciclo endógeno (P7). Co-movimiento contemporáneo dominante.
- Buildup de sobreacumulación ~4–5 trimestres antes de la recesión (P8) — re-confirma
  Tapia Granados (2012) con método independiente (estudio de evento).
- El feedback de maduración tiene **lag central ~1 año, identificable** (convergencia de
  CCF, distributed-lag, VAR(p≥4), jerárquico). Lo captura un **kernel de lag distribuido**.
- **Estabilidad:** el *centro* del lag es **estable entre crisis** (lag por crisis: media
  4.2, sd ~0.9 trim), pero la *fuerza* del acople es **no-estacionaria** (ventana móvil:
  −0.72 a +0.25). Y los parámetros de los osciladores LV/FN son inestables entre ciclos
  (P4, CV≈1). ⇒ lo estructural-estable es el lag, no la intensidad ni los params del oscilador.
- Modelo físico de **cadena de maduración** (P11): ODE diferenciable; el lag de
  maduración (~1 año) es un parámetro físico estimable.

**Descartado / callejones (NO re-intentar sin data nueva):**
- Oscilador full-series como ciclo límite → **degenera** (P1).
- Oscilador no lineal mejorando el pronóstico del **VAR** → no ocurre (P3); el VAR(p≥4)
  es, de hecho, un kernel de lag distribuido y captura el patrón.
- "El oscilador ajusta las crisis especialmente bien" → **trivial** (P9 placebo): ajusta
  cualquier ventana de 13 trim igual; el margen aparente es confound de amplitud.
- Osciladores de **fase fija** (LV/FN/VAR1) para el kernel → **refutados** (P10): su CCF
  es un coseno rígido, no el pulso+joroba observado.
- Identificar el **ANCHO** de la distribución del delay o su "fluctuación entre crisis"
  → **no identificable** con esta data (P10): SNR del feedback ~3%; el ancho se confunde
  con ruido por construcción. Lo identificable es el **centro** (~1 año), no la dispersión.
- PINN inversa con **estados de maduración latentes libres** → mal puesta (k→0, lag sin
  anclar). La correcta ata m = convolución del kernel con I (ver §5).

## 4. Nota metodológica crítica (no perder)

Las series principales venían en **YoY = `pct_change(4)`**, que por ser una diferencia a 4
trimestres **infla ~2×** el feedback en lag−4 (justo donde cae la señal). El transform
insesgado es **QoQ = `pct_change(1)`**. Todos los números de CCF/evento deben leerse en QoQ
(o con haircut ~2× sobre los YoY). El patrón **sobrevive** en QoQ, a menor magnitud.

## 5. Dónde está la frontera (lo más original)

El aporte más sólido y menos trillado es de **identificabilidad**: con ~9 crisis y un
feedback que explica ~3% de la varianza, se identifica el **centro** del lag de maduración
pero **no su dispersión**. Esto se caracteriza con un modelo físico diferenciable. La
calibración por **PINN inversa** (P11) usa la formulación de convolución del kernel Gamma
(la solución analítica de la cadena), que vuelve el inverso bien puesto — la versión con
latentes libres no lo es.

**Estado honesto del PINN (al cierre de la sesión; números en `results/p11_pinn_estimates.csv`):**
- **Sintético:** la convolución **identifica `k`** (k̂≈0.85 vs 1.1; la versión con latentes
  daba k→−0.1) pero **el lag se recupera con sesgo** (4 trim verdadero → 1.86, error 54%).
- **Datos reales:** **el lag COLAPSA a ~0** (0.23 trim) con parámetros no físicos
  (d=−2.0 = amortiguamiento negativo) y L_data alto → la red **no representa** la serie de
  145 trim (spectral bias en ventana larga) y el lag queda sin anclar.

⇒ **El PINN NO es la fuente del lag.** El lag central ~1 año está respaldado por los métodos
NO-PINN (CCF, distributed-lag, VAR(p≥4), filtro OLS de P11). El PINN es la calibración
diferenciable a afinar. **Cómo seguir:** (1) ajustar por **ventanas cortas por crisis**
(no 145 trim) para evitar spectral bias; (2) red con **features de Fourier**; (3) anclar
`k`/usar perfil de verosimilitud en μ; (4) recién con la recuperación sintética limpia,
leer el lag real.

## 6. Próximos pasos (priorizados)

1. **Cerrar P11 (PINN inversa, formulación de convolución):** corregir el **sesgo del lag**
   en la recuperación sintética (hoy ~50% error; k ya se identifica bien). Sospechas: el
   tradeoff residual k–μ y la discretización del kernel Gamma vs la cadena continua.
   Sugerencias: hacer coincidir exactamente el kernel con la solución de la cadena usada
   para generar; regularizar/anclar k; o estimar μ por perfil de verosimilitud (fijar μ,
   ajustar el resto, barrer μ). Recién con la recuperación sintética validada, leer el lag
   real. Resultados → `results/p11_pinn_estimates.csv`.
2. **Frontera de identificabilidad en sintético:** barrer n_episodios × longitud × SNR y
   medir cobertura del IC del lag — convierte el muro en resultado positivo.
3. **DDE diferenciable + capa jerárquica** (partial pooling del lag entre crisis) por
   autodiff/adjoint, reportando sensibilidad al prior del ancho.
4. **Diagnóstico formal de estabilidad del lag/kernel por crisis** (pieza dedicada, falta):
   estimar μ por episodio con IC + leave-one-out (¿el lag se sostiene al excluir cada
   crisis?). Es la validación análoga-a-holdout para un parámetro estructural. La data está
   lista (12 crisis con ventana completa, niveles para QoQ).

## 7. Reproducir / correr

```bash
uv sync                                            # entorno Python (uv.lock pinneado)
julia --project=julia -e 'using Pkg; Pkg.instantiate()'   # entorno Julia (Manifest pinneado)

uv run python -m src.experiment.p7_overaccumulation   # CCF (P7)
uv run python -m src.experiment.p8_event_study        # estudio de evento (P8)
uv run python -m src.experiment.p9_ablate             # ablation régimen de crisis (P9)
uv run python -m src.experiment.p10_varp              # bench VAR(p) kernel (P10)
uv run python -m src.experiment.p11_physical_delay    # identificabilidad del modelo físico
julia --project=julia src/experiment/p11_pinn_maturation.jl  # PINN inversa (P11)
```

`data/` está versionado (snapshot FRED), así que los números reproducen sin re-descargar.
Las figuras van a `output/experiment/` (gitignored, regenerable).
