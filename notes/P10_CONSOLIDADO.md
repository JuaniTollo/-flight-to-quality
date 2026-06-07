# P10 — ¿El feedback de sobreacumulación es un delay estocástico / kernel de lag distribuido?

**Transform principal:** QoQ = `pct_change(1)` sobre niveles (PROFITS, INVESTMENT); YoY solo como contraste (infla las magnitudes ~2×). **n = 9 crisis usables** (sin 2008/2020), 13 trim/ventana (±6 del fondo de inversión).

> Hipótesis evaluada: el feedback inversión-pasada → deprime-ganancia no es un retardo determinístico de τ fijo ni un oscilador de fase fija, sino un **delay estocástico / kernel de lag distribuido** cuyo τ es una variable aleatoria con distribución propia, que además **fluctúa entre crisis**.

---

## (a) Veredicto: parcial — un núcleo robusto y una parte fuerte no identificable

La hipótesis se descompone en tres claims independientes:

| Claim | Veredicto | Evidencia |
|---|---|---|
| **C1.** Existe un feedback negativo retardado con **lag central μ ≈ 4 trim**, separado del co-movimiento contemporáneo | **Se sostiene (robusto)** | Convergencia de 5 métodos: CCF trough lag4/5 = −0.31/−0.32; β empírico β4=−0.26, β5=−0.24; distlag pooled trough k=4; VAR(p≥4) k4/k5=−0.25/−0.26; jerárquico μ4=−0.20 (IC95 excluye 0). Trough por crisis: media 4.2, mediana 4. |
| **C2.** El kernel tiene **forma distribuida** (joroba en 4-5, vuelve a ~0 en lag6) | **Se sostiene como forma, pero no le gana al delay puntual** | La forma es real y convergente. Pero ΔAICc = −1.2 **prefiere el delay puntual τ=4**; el ancho `s` tiene perfil de verosimilitud plano (varía 1.8%, 2·ΔlogL=2.16 < χ²=3.84) y CI95(s)=[0,4]. **`s` no es identificable.** El pooled de 9 deltas determinísticos heterogéneos {τ=2,3,4,4,4,5,5,5,6} reproduce el kernel empírico → la "joroba distribuida" es observacionalmente equivalente a promediar deltas con τ fijo por crisis. |
| **C3.** El delay es **estocástico (distribución dentro de cada crisis)** y su distribución **fluctúa entre crisis** | **No se sostiene / no identificable** | (i) *Dentro:* simular una crisis con δ puntual recupera, por ruido sobre 13 pts, un ancho sd≈1.47, indistinguible de un kernel ancho (sd≈1.70). (ii) *Entre:* en el jerárquico la SD-entre-crisis de β_lag4 (el mecanismo central) = 0.001; β_lag4 ∈ [−0.219, −0.216] en las 11 crisis. La "fluctuación" de μ (rango [1,7]) es ruido de identificación: un μ constante=4 bajo este SNR genera sd≈1.4–1.6 por crisis. |

**El muro no es solo n=9 sino el SNR:** el feedback retardado explica **~3.2% de la varianza** de profits-z; el co-movimiento contemporáneo explica **~35%**. `s` modula la forma de ese 3% → confundido con ruido por construcción. Aun con 90 crisis `s` seguiría casi inestimable; un estimador Bayesiano lo regularizaría hacia el prior, no lo identificaría (shrinkage 0.92).

**Síntesis:** lo defendible es *"feedback de maduración débil pero detectable, centrado en ~4 trimestres (1 año), dominado por co-movimiento contemporáneo; su dispersión (dentro) y su variación entre crisis son no identificables con estos datos"*.

---

## (b) Ranking de modelos por captura del kernel

| # | Modelo | lag0 (+0.6) | joroba 4-5 | Identificable | Veredicto | Script |
|---|---|---|---|---|---|---|
| 1 | **VAR(p≥4-5)** (serie completa, n=312) | vía ρ residual (+0.46) ✓ | **sí, sin supervisión** (k4/k5=−0.25/−0.26) ✓ | buena | mejor baseline; **es por definición un kernel de lag distribuido libre** | `p10_varp.py` |
| 2 | **Distlag explícito** (Almon + libre) | β0=+0.56 ✓ | trough k=4, 51% masa neg en {4,5} ✓ | forma agregada robusta; coef individuales no sig | adecuado (descriptivo, con UQ) | `p10_distlag.py` |
| 3 | **Jerárquico (partial pooling)** | μ0=+0.57 ✓ | μ4=−0.20 sig ✓ | media sí; **dispersión τ_k no** (colapsa) | parcial | `p10_hier.py` |
| 4 | **Stochdelay (Gamma μ,s)** | b=+0.11 ✓ | μ*=4.40 ✓ | μ parcialmente; **`s` no** (perfil plano) | parcial→inadecuado para "es distribución" | `p10_stochdelay.py` |
| 5 | **FN (fase fija)** | +0.51 parcial | trough único mal ubicado (lag3); kernel alterna signo ✗ | re-tuneando 3 params/ventana | inadecuado: CCF = coseno, no pulso+joroba | `p10_lv_fn.py` |
| 6 | **VAR(1)/LIN (oscilador lineal 2D)** | sí (+0.49/+0.71) ✓ | **cero** en 4-5 (kernel geométrico, half-life <1 trim) ✗ | — | inadecuado: memoria demasiado corta | `p10_lin_var1.py` |
| 7 | **LV (fase fija)** | CCF plana ~+0.15 ✗ | sin estructura; kernel explota ✗ | no identificable | inadecuado + degenerado | `p10_lv_fn.py` |

**Mejor modelo:** VAR(p≥4-5) como baseline objetiva + distlag/jerárquico como representación interpretable con UQ. Los osciladores de **fase fija (LV/FN/VAR1) quedan refutados**: codifican el feedback como un retardo determinístico ≈T/2 atado al periodo (CCF = un solo coseno con trough rígido), incompatible con un pulso aislado en 0 + banda negativa en 4-5 que vuelve a 0. **Ese descarte es el resultado negativo limpio.**

---

## (c) Distribución entre crisis vs dentro de cada crisis

- **Dentro de cada crisis** (¿el τ de una crisis es un punto o una distribución?): **no identificable.** Con 13 pts ruidosos una crisis ≈ una sola oscilación; separar "lag medio" de "ancho del lag" requiere repetir la respuesta de impulso, y esa información no existe. Simulación: δ puntual recupera ancho sd≈1.47 ≈ kernel ancho sd≈1.70.
- **Entre crisis** (¿el lag central por crisis varía de episodio a episodio?): es lo único con algún contenido, pero frágil. Trough por crisis (QoQ) = {1954:−5, 1958:−4, 1961:−4, 1970:−5, 1975:−2, 1980:−4, 1982:−5, 1991:−4, 2001:−5} → media 4.2, sd 0.9, rango [2,5]. Es dispersión **entre** crisis de un lag por-crisis bien definido, no dispersión **dentro**. Pero el jerárquico da SD-entre-crisis del mecanismo (β_lag4) = 0.001 (las crisis comparten el mismo feedback), y `distlag` (sd trough=0.42) contradice a `stochdelay` (sd μ=2.27) → esa contradicción es la firma de no-identificabilidad.

**Separar δ-entre-crisis de distribución-dentro solo sería posible con varias respuestas de impulso por episodio (mayor frecuencia, o paneles sectoriales/multi-país). Es un límite duro de identificación.**

---

## (d) Qué es identificable y qué no (n=9, SNR≈0.057)

**Identificable y robusto:**
- Co-movimiento contemporáneo lag0 ≈ +0.6 (≈35% de varianza). El efecto dominante.
- Lag central del feedback μ ≈ 4 trim (1 año). Convergencia de 5 métodos.
- Magnitud del feedback (del FULL): β4 ≈ −0.25, β5 ≈ −0.24; CCF lag4/5 ≈ −0.31/−0.32.
- Forma cualitativa: pulso+ aislado en 0 + joroba− en 4-5 que vuelve a ~0 en lag6 (refuta el oscilador de fase fija).
- Contraste crisis-vs-expansión en QoQ (lag4: −0.29 crisis vs 0.00 expansión) → fenómeno de **régimen de crisis**, no de toda la serie.

**No identificable (retirar o marcar):**
- El ancho `s` del kernel (perfil plano, CI95=[0,4], AICc prefiere punto).
- μ/τ por crisis y "el delay fluctúa entre crisis" (SD β_lag4 entre crisis = 0.001).
- k4=−0.52/−0.60 del crisis-pooled (sobreajuste); citar el FULL ≈ −0.25.
- Distinción punto vs distribución dentro de una crisis.

**Tensión a declarar:** la joroba 4-5 solo aparece con VAR p≥4-5, justo donde BIC prefiere p=1. El patrón vive en zona que el criterio parsimonioso descarta (no fatal: 5 métodos convergen).

---

## (e) Resumen de resultados

Sobre el ensemble de 9 recesiones de posguerra (EE.UU., trimestral, QoQ), el feedback de sobreacumulación inversión→ganancia se representa como un **kernel de lag distribuido** —pulso contemporáneo positivo dominante + masa negativa centrada en ~4 trimestres— y un oscilador de fase fija (Lotka-Volterra / FitzHugh-Nagumo / VAR(1)) **no puede generar esa forma** porque impone un retardo determinístico atado al periodo. El resultado central es de **identificabilidad**: con n=9 crisis y un feedback que explica ~3% de la varianza, el **lag central (~4 trim) y la forma del kernel son robustos y convergen entre 5 métodos**, pero el **ancho de la distribución del delay y su variación entre crisis son estructuralmente inestimables** (perfil de verosimilitud plano en `s`, AICc favorece el delay puntual, shrinkage jerárquico 0.92, dispersión entre crisis del mecanismo ≈ 0). El estimador jerárquico cuantifica cuánto no se puede saber.

**Relación con la teoría:** el lag medio ~4-5 trim coincide con Tapia Granados (2012), que documenta que la ganancia cae 4-5 trimestres antes de la recesión. "Distribución del lag" y "fluctúa entre crisis" no están en Tapia. El microfundamento de un kernel distribuido es la **agregación de proyectos de inversión heterogéneos con distintos lags de gestación** (Kalecki 1935; Kydland-Prescott 1982): un kernel distribuido es la consecuencia mecánica de agregar. Referencias: Kalecki/Kydland-Prescott (agregación), Tapia Granados (lag medio), Almon (1965, lag distribuido).

---

## (f) Trabajo futuro

1. **Recuperación en sintético calibrada al SNR real.** Generar ensembles desde un kernel conocido (μ*, s*, τ*) con 9 episodios × 13 pts y σ≈0.73; estimar con el pipeline jerárquico; medir cobertura del IC95. Barrer n_episodios={9,20,50,100} × longitud={13,26,52} → frontera de identificabilidad.
2. **Placebo del kernel + CCF-Lyapunov del VAR(p≥4).** (a) ¿La joroba 4-5 sobrevive en ventanas de expansión / aleatorias? (b) Derivar la CCF teórica vía Lyapunov de un VAR(p≥4) con los λ estimados; si reproduce la joroba, "kernel de delay" ≈ "lineal estocástico de memoria larga" en la CCF de 2º orden.
3. **Kernel/DDE como modelo diferenciable.** Convolución de lag distribuido o DDE con kernel Gamma(μ,s), parámetros por autodiff + adjunto (Julia SciML), con capa jerárquica (partial pooling de (μ,s)), reportando sensibilidad al prior en `s` y el factor de Bayes distribución-vs-punto.
