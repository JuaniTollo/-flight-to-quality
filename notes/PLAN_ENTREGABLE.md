# Plan de entregable DM2026 — síntesis de la re-auditoría multiagente (2026-06-13)

Producto de un workflow de 20 agentes Opus (re-auditoría de 5 pilares + 7 caminos
alternativos + crítica adversarial + síntesis). Resumen ejecutable para cerrar la materia.

## Tesis (lo que SÍ se afirma, sin asteriscos)

**Analizar la hipótesis de Tapia con ODEs + PINNs** — y el resultado honesto es de
**identificabilidad**: con n=9 crisis y un feedback al ~3% de SNR, el **centro** del retardo
de maduración (~1 año / 4-5 trim) se identifica de forma convergente (5 métodos clásicos + un
PINN inverso validado en sintético al 8.9%) y **refuta los osciladores de fase fija**; pero su
**dispersión** y su **separabilidad de un VAR(p≥4)** NO se identifican. Esa frontera medida ES
la contribución, no una derrota.

## Pregunta que el trabajo responde afirmativamente

Dado el ciclo ganancias–inversión real (FRED trimestral 1948+), ¿qué se PUEDE saber de su
retardo estructural y con qué método inverso, cuando hay n=9 crisis y el feedback explica ~3%
de la varianza? → El centro del retardo (~1 año) es recuperable y convergente; el ancho, la
especificidad-de-crisis y la separabilidad de un VAR lineal no lo son; la forma del kernel
(pulso+ en lag0, lóbulo− en lag4-5) refuta los predador-presa de fase fija (Goldstein/LV/FN).

## La cadena Tapia → ODE → PINN

1. **Tapia (verbal):** la inversión, al madurar, deprime la ganancia con retardo ~1 año
   (+ firma pre-crisis: la ganancia cae 4-5 trim antes de la recesión).
2. **ODE (formalización):** cadena de maduración `dmⱼ/dt=(mⱼ₋₁−mⱼ)/θ` → por linear chain
   trick equivale a un kernel Gamma (retardo distribuido). Es la encarnación física de Tapia.
3. **PINN (problema inverso):** calibración diferenciable que estima el retardo μ optimizando
   conjuntamente la red sustituta y los parámetros físicos (Zygote/Adam, features de Fourier).
4. **Validación en sintético:** lag verdadero 4.0 → recuperado 4.36 (**error 8.9%**) desde un
   init lejano; 3 variantes clásicas colapsan/sesgan ~53-59% por spectral bias.
5. **Datos reales:** ~1 año, **consistente con** los 5 métodos no-PINN — NO una medición
   (R²(red)=1 = interpolación; el término −k·m casi no restringe).

## Resultados positivos (presentables)

- Co-movimiento contemporáneo dominante (lag0 ≈ +0.6, ~35% var).
- Retardo negativo centrado en ~1 año, convergente en 5 métodos, estable al LOO.
- **Refutación limpia de los osciladores de fase fija** (resultado negativo más distintivo).
- Re-confirmación independiente de Tapia 2012 (event-study NBER: buildup pre-valle).
- **PINN-Fourier validado en sintético (8.9%)** — resultado de método (scientific ML).
- Frontera de identificabilidad (centro sí, ancho/VAR no) — el aporte original y transferible.

## Honestidad obligatoria (caveats a declarar)

- n=9 EXCLUYE 2008 y 2020 — declararlo en cada claim (anticipa "¿y 2008?").
- R²=1.00 real = interpolación de la red sustituta (~2160 pesos / ~290 valores), NO validación
  física. La validez del método viene del sintético + corroboración clásica.
- El lag real 3.90 trim es "consistente con", nunca "medido por el PINN".
- −k·m NO explica la caída de ganancias: OLS directo da β_m=+0.020 (signo contrario, t=0.15,
  ΔR²≈0) por colinealidad m~I. No se afirma mecanismo causal.
- El kernel ≡ VAR(p≥4) (Lyapunov): re-parametrización interpretable, no mecanismo extra.
- El lóbulo lag4-5 es UBICUO (placebo p=0.34): estructural, no firma de crisis. El placebo NO
  refuta a Tapia (Tapia nunca comprometió especificidad de crisis del kernel).
- Lag = "~1 año (4-5 trim)", sensible al transform (QoQ→4, YoY→5 inflado ~2x). Reportar ambos.
- Atribución: Goodwin 1967 = método (LV sobre empleo/salario), Goldstein 1999 = LV de fase
  fija (único target de la refutación de 90°), Tapia = correlaciones direccionales retardadas.

## Crisis (deseo "al menos modelar las crisis")

Cumplido SOLO de forma **descriptiva** vía el event-study NBER (P8): el buildup de la brecha
inversión−ganancia pre-valle (gap significativo en τ=−6..−4) es crisis-específico y limpio, y
reconfirma Tapia 2012. NO intentar modelo causal de crisis ni régimen dinámico especial: dos
placebos (p9_placebo, p12_kernel_placebo p=0.34) muestran que las crisis no son un régimen
separable. El PINN per-crisis falla en sintético (59%) — no usarlo.

## Plan de la semana (entrega 19/06, poster 17/06)

- **Día 1 — limpieza (CORE, 0 compute):** borrar `results/p12_pinn_profile*.csv` (smoke
  abortado, contradice el headline). Corregir `README.md` raíz L3-6 (atribución LV). Borrar de
  STATUS/EXPERIMENT la nota "los .jl pueden no existir" (FALSA: los 5 existen).
- **Día 1-2 — edición del paper (CORE, 0 compute, alto valor):** elevar la pregunta de
  identificabilidad al primer párrafo de la Intro; caja "Dos lecturas del término de maduración
  y por qué no coinciden" (k_PINN=+0.42 junto a β_m_OLS=+0.020); mencionar Lux.jl/
  DifferentialEquations.jl con versión + licencia MIT en Datos y software.
- **Día 2 — FIX #1 (CORE, ~1 línea Julia):** guardar `L_phys` (residuo físico, ya computado en
  `comps()` pero descartado) además de R² en el CSV real. Convierte "R²=1 es interpolación" de
  disculpa en evidencia cuantitativa.
- **Día 2-3 — logística (CORE):** repo público y pusheado, licencia MIT, link OK. Issue de
  GitHub con etiqueta "validado" + su NÚMERO (va en el asunto del mail). Conteo de páginas
  (7-12 sin refs/apéndices; hoy 7 — cumple).
- **Día 3-4 — DEFENSA ORAL del poster (CORE, el diferenciador de nota):** ensayar respuestas a
  "¿por qué un VAR(p≥4) no te invalida?", "¿por qué reportás un PINN con R²=1?", "¿y 2008/2020?".
- **Día 4-5 — recovery curve sintética (OPCIONAL, único compute de horas):** barrer
  μ_true ∈ {2,6,8} (el 4 ya existe) + figura estimado-vs-verdadero (recta y=x). CRÍTICO:
  parametrizar el init como verdadero±offset constante (si no, μ=8 con init=8 es trivial).
- **Día 5-6 — oscilador lineal amortiguado (OPCIONAL, sacrificable):** ajustar 2º orden a la
  crisis compuesta como sustanciación cuantitativa del bajo amortiguamiento. Solo si sobra.

## Posición de repliegue (si un evaluador objeta "todo se descarta")

El trabajo NUNCA afirma haber probado el mecanismo causal. Afirma, y eso sobrevive: (1) un
retardo estructural de ~1 año recuperado por 5 métodos, (2) la refutación de los osciladores de
fase fija por la forma del kernel, (3) un inverso diferenciable validado en sintético (8.9%),
(4) la ubicación medida de la frontera de identificabilidad. "Un VAR lo reproduce" → sí, y lo
declaramos: ese ES el resultado de identificabilidad; el VAR(1)/fase-fija sí queda refutado.
"R²=1 es sobreajuste" → sí, lo marcamos; la validez viene del sintético + métodos clásicos. El
criterio de la materia (solidez metodológica, capacidad de responder preguntas) premia
exactamente este ejercicio de honestidad de frontera.
