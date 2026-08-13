# La cadena de retardo, paso a paso

Nota de estudio: qué dice exactamente `dmⱼ/dt = (mⱼ₋₁ − mⱼ)/θ`, de dónde sale,
y cómo se conecta con el núcleo Gamma y la frontera de identificabilidad.

## 1. Qué es cada símbolo

- `mⱼ(t)`: cuánta inversión hay *dentro* de la etapa `j` en el momento `t`.
  El índice `j` **no es tiempo**: es la posición en la fila de maduración
  (1 = en obra, 2 = capacidad nueva, …, N = saturando el mercado).
- `dmⱼ/dt`: a qué velocidad cambia ese contenido. Positivo = la etapa se llena;
  negativo = se vacía.
- `θ`: tiempo medio de residencia en cada etapa (en trimestres).
- `m₀ = I(t)`: la primera etapa se alimenta de la inversión corriente.
- `m(t) = m_N(t)`: la salida de la última etapa es la inversión **madurada**,
  la única que toca la ganancia (vía `+k·m`, con `k < 0`).

## 2. De dónde sale la ecuación: balance de tanque

El contenido de una caja cambia por lo que entra menos lo que sale:

- **entra** lo que drena la caja anterior: `mⱼ₋₁/θ`
- **sale** lo que drena hacia la siguiente: `mⱼ/θ`

La regla de drenaje es "cada caja pierde en proporción a lo que tiene, a ritmo
`1/θ`" — un tanque con un agujero: cuanto más lleno, más rápido drena, y una
gota pasa en promedio un tiempo `θ` adentro. Entonces:

```
dmⱼ/dt = mⱼ₋₁/θ − mⱼ/θ = (mⱼ₋₁ − mⱼ)/θ
```

La resta es literalmente *entrada menos salida* con `θ` como factor común.

**Lectura por signo:** si `mⱼ₋₁ > mⱼ`, la derivada es positiva y la caja se
llena; si es menor, se vacía. Cada etapa **persigue el nivel de la anterior**,
y `θ` es cuánto tarda en alcanzarla. Por eso "suavizado exponencial": es la
misma matemática del promedio móvil exponencial.

## 3. Versión discreta: cómo calcular t+1 sabiendo t

Estado en `t`: `P(t)`, `I(t)`, `m₁(t), …, m_N(t)`. Con paso de 1 trimestre:

```
mⱼ(t+1) = mⱼ(t) + (mⱼ₋₁(t) − mⱼ(t))/θ        j = 1, …, N   (m₀ = I)

P(t+1)  = P(t) + c·I(t) − d·P(t) + k·m_N(t)
I(t+1)  = I(t) + a·P(t) − b·I(t)
```

El efecto retardado entra por un solo lugar: `k·m_N(t)`, lo que hay hoy en la
última caja. El resto de la cadena es contabilidad interna para que eso sea,
efectivamente, inversión vieja.

Agrupando la ecuación de las cajas:

```
mⱼ(t+1) = (1 − 1/θ)·mⱼ(t)  +  (1/θ)·mⱼ₋₁(t)
          └── se queda ──┘     └─ llega de atrás ─┘
```

Cada trimestre la caja retiene una fracción `(1 − 1/θ)` de lo suyo y recibe
una fracción `1/θ` de la caja anterior.

## 4. Los dos casos extremos

- **θ = 1**: queda `mⱼ(t+1) = mⱼ₋₁(t)` — cada caja le pasa *todo* a la
  siguiente, cada trimestre. Cinta transportadora exacta: `m_N(t) = I(t−N)`,
  retardo **puntual** de N trimestres clavados. (La intuición "en el trimestre
  t = j el peso está en la caja j" es exactamente este caso.)
- **θ = 2**: queda `mⱼ(t+1) = ½·mⱼ(t) + ½·mⱼ₋₁(t)` — la caja se queda con la
  mitad y recibe la mitad. Cada paso **mezcla**; tras N cajas la mezcla
  acumulada es una campana: mayormente inversión de hace `μ = N·θ` trimestres,
  con colas a los costados.

El único lugar donde se fabrica el desparramo es el término `(1 − 1/θ)`:
si es cero → retardo puntual; si es positivo → retardo distribuido.

## 5. Por qué el núcleo es exactamente Gamma

Cada etapa retiene un tiempo aleatorio ~ Exponencial(θ). El tiempo total de
tránsito es la suma de N exponenciales independientes, que es **exactamente**
una Gamma/Erlang (truco de la cadena lineal, Smith 2011):

```
w(t) ∝ t^(N−1) · e^(−t/θ)        centro μ = N·θ,  ancho √N·θ
```

El núcleo Gamma no se elige por conveniencia: **emerge** de suponer etapas en
serie. Un pulso de inversión en t = 0 sale por `m_N` como una lomada centrada
en `μ`, no como un espigón.

## 6. Conexión con la frontera de identificabilidad

La pregunta "¿puedo suponer que el efecto llega exacto a los N trimestres?"
es matemáticamente la pregunta "¿el núcleo es un espigón puntual o una campana
ancha?". El resultado central es que **el dato no las distingue**: el ajuste
fija el centro (`μ ≈ 5` trimestres) pero es plano frente al ancho — la cinta
exacta `(θ=1, N=5)` y combinaciones más mezcladoras con el mismo `μ` ajustan
igual. La cadena no está para complicar la cinta transportadora: está para que
el *ancho* del retardo sea un parámetro del modelo y poder preguntarle al dato
si lo identifica. La respuesta (no) es tan resultado como el `μ ≈ 5`.

## 7. Ejercicio de verificación (5 líneas)

Pulso de inversión en t = 0, θ = 2, N = 3: el pulso entra concentrado y sale
por `m₃` como una lomada centrada en t ≈ 6.

```python
import numpy as np
theta, N, T = 2.0, 3, 30
m = np.zeros(N); I = 100.0
for t in range(T):
    m0 = I if t == 0 else 0.0
    entrada = np.concatenate([[m0], m[:-1]])
    m = m + (entrada - m) / theta
    print(t, np.round(m, 2))
```
