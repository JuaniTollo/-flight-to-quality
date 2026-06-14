# P12 — Placebos de falsación del mecanismo de sobreacumulación

Registro de los tests de falsación que distinguen un **mecanismo real** de un **artefacto de
fondo**. Complementa `EXPERIMENT.md` y `notes/STATUS.md` §1.

## Pregunta

El feedback de sobreacumulación (inversión madurada → deprime ganancia con retardo ~1 año)
aparece como una **joroba negativa del kernel de lag distribuido / CCF en los rezagos 4-5**.
¿Esa joroba es **propia del régimen de crisis** o es **ubicua** (parte de la dinámica de
fondo de las series, no una firma del giro del ciclo)?

## p12_kernel_placebo — RESULTADO (corrida completa, 2026-06-09)

Estadístico de la joroba (lags 4-5, QoQ, z-score por ventana) en tres conjuntos de ventanas:
crisis (anclas NBER), expansión (centros entre crisis), aleatorias no-crisis (n=156).

```
VEREDICTO
  KERNEL β_k:  joroba crisis (POOLED) = -0.181, percentil 47 en placebo → UBICUA
  CCF:         joroba crisis (POOLED) = -0.215, percentil 27 en placebo → UBICUA
```

| estadístico | crisis (pooled) | aleatorio (mediana) | Mann-Whitney crisis<aleatorio |
|---|---|---|---|
| KERNEL β₄₅ | −0.181 | −0.173 | p = 0.341 (no sig.) |
| CCF lag 4-5 | −0.215 | −0.149 | p = 0.065 (marginal) |

**Conclusión:** la joroba negativa de sobreacumulación en lag 4-5 **NO es propia de las
crisis** — aparece igual de fuerte fuera de ellas. (El smoke había dado el CCF "concentrado"
con p=0.031, pero con la muestra completa de 156 placebos se diluye a 0.065.)

Figura: `output/experiment/p12_kernel_placebo.png`.

## Implicancia para el claim

Hay que separar dos versiones de la hipótesis:

- **Versión fuerte (FALSADA):** "la sobreacumulación es una *firma del giro del ciclo*, se
  enciende en las crisis." → **No.** El efecto es igual de fuerte fuera de las crisis.
  Esto matiza la frase de `STATUS.md` §1 ("más intensa cerca de las crisis"): el contraste
  crisis-vs-expansión **no sobrevive** al placebo formal del estadístico del kernel.
- **Versión estructural (EN PIE):** "existe un feedback de sobreacumulación con retardo:
  la inversión madurada (~1 año atrás) deprime las ganancias, `k>0`." → Sobrevive, y el
  placebo la refuerza: la joroba negativa lag 4-5 está presente **siempre** (crisis,
  expansión y aleatorio dan todos mediana negativa). Es un rasgo **estructural/permanente**
  de la relación profits–investment, no un artefacto de ventana ni un fenómeno episódico.

→ El mecanismo de maduración-que-deprime es **continuo**, no exclusivo de las recesiones.
Lo defendible: "la inversión madura y deprime la ganancia con ~1 año de retardo, de forma
estructural". Lo NO defendible: "esto explica/dispara las crisis".

## p12_overaccum_decomp — ¿−k·m explica la caída de ganancias? (2026-06-09)

Descomposición OLS jerárquica de la ecuación del modelo (`dP/dt = c·I − d·P − k·m`) sobre
los datos reales (QoQ z-score, ventana 1990, igual que el PINN Fourier). Mide el aporte
marginal del término de sobreacumulación por encima del co-movimiento contemporáneo.

```
M0  dP ~ I,P     : R² = 0.094        M1  dP ~ I,P,m(μ=3.9) : R² = 0.094  (ΔR² = +0.000)
β_m = +0.020 (t=0.15)  → signo POSITIVO (contrario al esperado), NO significativo
peor decil de caídas: la sobreacumulación aporta 0% de la caída media
```

Robusto al suavizado (SavGol w=5/9/13): ΔR² ∈ [0.0006, 0.022] y **β_m POSITIVO en todos**
los casos. **El k=0.42>0 del PINN Fourier NO se reproduce en una atribución directa.**
Causa probable: `m` (inversión madurada) es colineal con `I` contemporánea → el canal de
maduración no es separable del co-movimiento contemporáneo. Script:
`src/experiment/p12_overaccum_decomp.py`.

## p12_var_lyapunov — la joroba es observacionalmente equivalente a un VAR lineal (2026-06-09)

¿La joroba negativa de lag 4-5 es un kernel de delay genuino o lo que un VAR(p) lineal
produce por construcción? Se deriva la CCF teórica vía Lyapunov y se compara con la empírica:

```
VAR(4): joroba teórica mín −0.224 @ lag−4  (empírica −0.231 @ lag−4) → REPRODUCE
VAR(5), VAR(6): idem. VAR(2/3): no.
=> EQUIVALENCIA OBSERVACIONAL (a nivel CCF de 2º orden)
```

**Un VAR(p≥4) lineal estacionario YA genera la joroba sin postular kernel de delay.** El
"kernel de maduración" es una re-parametrización interpretable del mismo contenido lineal,
**no un mecanismo adicional identificable** con estos datos.

## p12_lag_stability — el lag solo emerge al poolear y es débil (2026-06-09)

Spec `P_t ~ c·I_t + e·P_{t-1} + k·m(μ)`, μ por crisis con IC + leave-one-crisis-out.

```
μ por crisis: ruidoso (1.0–5.7), perfiles de R² casi planos (se mueven ~0.08 al variar μ)
μ pooled = 4.72 trim, IC95 = [1.0, 5.74] (ancho) | LOO estable (4.33–4.96)
```

El pooled es estable al LOO (no depende de un episodio), PERO el perfil es casi plano: el
R²=0.42 viene del co-movimiento contemporáneo + persistencia, NO del término de maduración
(aporta ~3%). Consistente con la descomposición.

## SÍNTESIS — qué se sostiene y qué no (2026-06-09)

| Afirmación | Veredicto |
|---|---|
| Co-movimiento contemporáneo profits–investment fuerte (lag0 ≈ +0.5) + persistencia | ✅ robusto |
| Joroba negativa lag 4-5 como **patrón descriptivo**, estable entre crisis (LOO) | ✅ robusto |
| Joroba = **mecanismo de maduración/sobreacumulación causal** distinto de dinámica lineal | ❌ no identificable (VAR equiv.) |
| El término −k·m explica la **caída** de ganancias | ❌ ΔR²≈0, signo contrario, colineal con I |
| Sobreacumulación como **firma de crisis** (vs expansión) | ❌ ubicua (kernel_placebo) |

**Defendible:** profits–investment forman un sistema lineal estocástico de memoria larga con
co-movimiento contemporáneo dominante y una firma de retardo de ~1 año robusta y estable.
**NO defendible como probado:** un mecanismo físico de sobreacumulación madurada distinto e
identificable por encima de esa dinámica lineal. El modelo de maduración es una re-descripción
interpretable, no un mecanismo identificado con estos datos.

## p12_pooled_placebo — DETENIDO (no aportaba a estos problemas)

Segundo ángulo: ¿la *dinámica compartida* (pooling de un único θ entre crisis, de P9) es
**especial** o **trivial**? Lanzado full en background (n_sets=150, budget 75s/ajuste, 2
modelos LV/FN) → `results/p12_pooled_placebo.log`. p chico → pooling de crisis especial;
p ≈ 0.5 → trivial (las crisis poolean como cualquier conjunto de oscilaciones cortas).

Cómo correr: `PYTHONPATH=. .venv/bin/python -m src.experiment.p12_pooled_placebo`
(smoke: `--smoke`; el smoke de 4 sets ya excede ~400s — el full es de horas).
