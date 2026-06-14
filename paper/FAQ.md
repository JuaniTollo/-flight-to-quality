# FAQ — Correlación, significancia y CCF de un VAR

## 1. ¿Cómo se mide la correlación?

La correlación de Pearson entre dos variables `x` e `y` es la covarianza
normalizada:

```
              Σ_t (x_t − x̄)(y_t − ȳ)
  r  =  ──────────────────────────────────────────
         sqrt( Σ_t (x_t − x̄)² ) · sqrt( Σ_t (y_t − ȳ)² )
```

- **Numerador** (covarianza): ¿cuando `x` está por encima de su media, `y`
  también tiende a estarlo?
- **Denominador** (las dos desviaciones estándar): divide para que el
  resultado quede **adimensional y acotado en `[-1, +1]`**.

El **signo** indica la dirección; el **valor absoluto** `|r|` indica la fuerza:

| Valor   | Interpretación                    |
|---------|-----------------------------------|
| r = +1  | relación lineal perfecta positiva |
| r = −1  | relación lineal perfecta negativa |
| r = 0   | sin relación lineal               |

> **Clave:** `r = −1` es *tan fuerte* como `r = +1`. Una es perfecta hacia
> abajo, la otra hacia arriba. La fuerza la da `|r|`, no el signo.

---

## 2. ¿Por qué una correlación de 0.3 puede ser "significativa"?

**Significativo ≠ grande.** Significa: *es improbable que un `r` de este tamaño
haya salido por puro azar si la correlación verdadera fuera 0.*

Eso depende del **tamaño de muestra `n`**. El estadístico de prueba es:

```
  t  =  r · sqrt( (n − 2) / (1 − r²) )
```

Mirá cómo entra `n` con el mismo `r = 0.3`:

| n     | t       | ¿Significativo?       |
|-------|---------|-----------------------|
| 10    | ≈ 0.89  | No (podría ser ruido) |
| 1000  | ≈ 9.9   | Sí (p casi cero)      |

Por eso "`r = 0.3` es significativo" **solo tiene sentido junto con `n`**. Con
muchos datos, hasta correlaciones chicas pasan el test, porque hay evidencia
suficiente para descartar que el valor verdadero sea exactamente 0.

> **En una CCF (series temporales):** regla rápida → banda `±2/sqrt(n)`. Un
> coeficiente de correlación cruzada se considera significativo si sale de esa
> banda. Con `n = 400`, la banda es `±0.1`, así que un `0.3` la cruza
> holgadamente.

---

## 3. ¿Por qué un −1 no es "más significativo"?

Dos correcciones a la confusión típica:

**a) "−1 positivo" no existe.** `−1` es una correlación **perfecta negativa**.
El signo menos *es* la dirección. No hay tal cosa como "correlación positiva
de −1".

**b) Fuerza ≠ significancia.** Un `r = −1` (o `+1`) es la relación más fuerte
posible, pero "fuerte" y "significativo" responden preguntas distintas:

|                       | Pregunta que responde                | De qué depende              |
|-----------------------|--------------------------------------|-----------------------------|
| **Fuerza** (`\|r\|`)  | ¿Qué tan apretada es la relación?    | Solo de los datos, no de n  |
| **Significancia**     | ¿Podría ser azar?                    | De `r` **y** de `n`         |

Un `r = −1` con `n = 3` puntos puede ser **menos significativo** (más fácil de
obtener por azar) que un `r = 0.3` con `n = 10000`. Tres puntos casi siempre
caen casi en una recta de casualidad; ahí un `|r|` enorme no prueba gran cosa.

### Resumen mental

- **Signo** → dirección.
- **`|r|`** → fuerza de la relación.
- **Significancia** → ¿es distinguible de cero dado cuántos datos tengo? Sube
  con `n`.

Un `0.3` con muchos datos puede ser significativo aunque sea una relación
débil. Un `−1` es la relación más fuerte que existe, pero su significancia
todavía depende de cuántos puntos la sostienen. Por eso **"más fuerte" no
implica automáticamente "más significativo"**.

---

## 4. La ecuación de Lyapunov y la CCF teórica de un VAR

### La idea de fondo

Es una **prueba de nulidad** (*null model*): antes de afirmar que tu mecanismo
(causalidad, retroalimentación, *flight to quality*, etc.) explica el patrón de
correlaciones cruzadas que ves en los datos, comprobás si un modelo **lineal
genérico sin ese mecanismo** —un VAR(p)— ya lo reproduce solo. Si lo reproduce,
el patrón **no** es evidencia de tu mecanismo.

Para eso necesitás la CCF *teórica* del VAR, y ahí entra Lyapunov.

### Qué es un VAR(1)

```
  x_t  =  A · x_{t−1}  +  ε_t
```

- `x_t` es un vector (`k` variables apiladas),
- `A` es la matriz `k×k` de coeficientes (cada variable depende del rezago de
  todas),
- `ε_t` es ruido con covarianza `Σ` (innovaciones), es decir `E[ε_t · ε_tᵀ] = Σ`.

### La ecuación de Lyapunov

Querés la **covarianza contemporánea** del proceso, `Γ₀ = E[x_t · x_tᵀ]`.
Tomando covarianza a ambos lados del VAR(1):

```
  Γ₀  =  A · Γ₀ · Aᵀ  +  Σ
```

Esta es la **ecuación de Lyapunov discreta** (a veces "ecuación de Stein"). Es
una ecuación matricial cuya **incógnita es `Γ₀`**. Dice: la varianza total del
sistema es la varianza propagada un paso (`A·Γ₀·Aᵀ`) más la inyección de ruido
nuevo (`Σ`).

Intuición de cada término:

- `A·Γ₀·Aᵀ` → cómo la dinámica **recicla** la varianza que ya había.
- `Σ`       → **energía nueva** que entra en cada período.

Existe solución única y estable si **todos los autovalores de `A` están dentro
del círculo unitario** (proceso estacionario). No hace falta resolverla a mano:
`scipy.linalg.solve_discrete_lyapunov(A, Sigma)` devuelve `Γ₀` directamente
(ver el snippet de Python más abajo).[^1]

[^1]: La solución cerrada se obtiene vectorizando la ecuación; es solo el
"cómo se resuelve por dentro" y no hace falta para usarla.

### De Γ₀ a la CCF

Con `Γ₀`, las **autocovarianzas a rezago `h`** salen por recursión:

```
  Γ_h  =  A · Γ_{h−1}  =  Aʰ · Γ₀        (h ≥ 0)
```

Y la **CCF teórica** entre la variable `i` y la `j` es `Γ_h` normalizada:

```
                  Γ_h[i, j]
  ρ_ij(h)  =  ───────────────────────────
               sqrt( Γ₀[i,i] · Γ₀[j,j] )
```

Esa `ρ_ij(h)` como función de `h` es la curva que comparás contra la CCF
empírica de tus datos.

### Para VAR(p)

No cambia la idea, solo el armado: reescribís el VAR(p) en **forma de
compañía** (*companion form*), un VAR(1) gigante donde el estado apila los `p`
últimos vectores:

```
  X_t  =  Ã · X_{t−1}  +  E_t

                ⎡ x_t      ⎤
                ⎢ x_{t−1}  ⎥
        X_t  =  ⎢   ⋮      ⎥
                ⎣ x_{t−p+1}⎦
```

Aplicás Lyapunov a `Ã` y `Σ̃`, obtenés la `Γ₀` del sistema aumentado, y los
bloques de arriba a la izquierda te dan las autocovarianzas del VAR(p)
original.

### El argumento completo

1. Ajustás un VAR(p) a tus datos → obtenés `Â`, `Σ̂`.
2. Resolvés Lyapunov → CCF teórica *implícita* en ese modelo lineal.
3. La comparás con la CCF empírica (y/o con la que predice tu modelo con
   mecanismo).
4. **Si el VAR sin mecanismo ya reproduce el patrón** (picos, asimetría, rezago
   del máximo) → ese patrón es un artefacto de dependencia lineal /
   autocorrelación, **no** prueba tu mecanismo. **Si el VAR no lo reproduce** →
   ahí sí tenés señal de algo que el lineal estándar no captura.

> En esencia: el VAR es la hipótesis nula *"todo es lineal y sin estructura
> especial"*, y Lyapunov es la herramienta que da **analíticamente** —sin
> simular— qué correlaciones cruzadas predeciría esa nula.

### Cálculo en Python

```python
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

# A: matriz de coeficientes del VAR(1) (o forma de compañía del VAR(p))
# Sigma: covarianza de las innovaciones
Gamma0 = solve_discrete_lyapunov(A, Sigma)

# autocovarianza a rezago h y CCF normalizada entre las variables i, j
def ccf(i, j, h, A, Gamma0):
    Gamma_h = np.linalg.matrix_power(A, h) @ Gamma0
    return Gamma_h[i, j] / np.sqrt(Gamma0[i, i] * Gamma0[j, j])
```
