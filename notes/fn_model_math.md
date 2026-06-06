# Matemática del modelo ganador — FitzHugh-Nagumo jerárquico + pronóstico condicionado

Referencia de las ecuaciones tal como están implementadas en
`src/experiment/fn_hierarchical.jl` y `src/experiment/fn_forecast.jl`.

---

## 1. El oscilador: FitzHugh-Nagumo (FN)

Sistema dinámico 2D de **relajación** con un **ciclo límite estable**:

$$
\dot V = c\Big(V - \tfrac{V^3}{3} - R + I\Big), \qquad
\dot R = \frac{V - a - b\,R}{c}
$$

- $V(t)$ = **profits** (activador, variable rápida).
- $R(t)$ = **investment** (recuperación / inhibidor, variable lenta).
- Parámetros $\theta = (a, b, c, I)$. En el código se usa $c = e^{\,\ell}$ con
  $\ell = \log c$ como parámetro libre, garantizando $c>0$ (escala temporal).
  Entonces internamente $\theta = (a, b, \ell, I)$.

**Lectura económica:**
- $V$ rápido, $R$ lento $\Rightarrow$ **los profits lideran a la inversión** (Tapia).
- El acoplamiento $-R$ en $\dot V$: la inversión acumulada **erosiona la ganancia**
  (sobreacumulación). El término cúbico $-V^3/3$ es la **saturación**.
- Asimetría boom-lento / crash-rápido $=$ dinámica de relajación.

**Por qué FN y no Lotka-Volterra.** El LV
$\dot x=\alpha x-\beta xy,\ \dot y=\delta xy-\gamma y$ tiene una **cantidad conservada**
$\Rightarrow$ órbitas **neutralmente estables**: la amplitud la fija la condición
inicial y no hay atractor. FN tiene **ciclo límite** (amplitud preferida a la que el
sistema vuelve tras un shock) y forma asimétrica. Empíricamente FN explicó los profits
(R²≈42% vs ~0 del LV).

---

## 2. Representación de los datos (anti-leakage)

Series $\text{YoY}$ de profits $P_t$ e investment $J_t$. Se estandarizan con media y
desvío calculados **solo con datos de entrenamiento** ($t \le$ 2001, sin tocar test):

$$
V_t = \frac{P_t - \mu_P}{\sigma_P}, \qquad
R_t = \frac{J_t - \mu_J}{\sigma_J}.
$$

FN vive **centrado en 0**, así que no hace falta el *offset* de positividad que el LV
exigía. La inversión a unidades YoY es $P = V\sigma_P + \mu_P$ (ídem $J$).

**Grupos / ciclos.** Cada ciclo $k$ es la ventana entre valles NBER consecutivos
$(\text{valle}_{k-1}, \text{valle}_k]$. Tiempo $t$ en años desde el inicio del ciclo;
condición inicial $u_{0,k} = (V,R)$ del primer trimestre del ciclo (fija).

---

## 3. Calibración jerárquica Bayesiana (el "pool")

Modelo generativo con **pooling parcial** sobre los ciclos de entrenamiento
$k = 1,\dots,K$:

$$
\begin{aligned}
\mu &\sim \mathcal N\!\big(m_0,\ \Sigma_0\big) && \text{media global de } (a,b,\ell,I)\\
\tau_j &\sim \text{HalfNormal}(0,\,0.4),\ j=1\dots4 && \text{dispersión entre ciclos}\\
\sigma &\sim \text{HalfNormal}(0,\,0.5) && \text{ruido de observación}\\[2pt]
\theta_k &\sim \mathcal N\!\big(\mu,\ \operatorname{diag}(\tau^2)\big) && \text{parámetros del ciclo } k\\
u_k(t) &= \text{solución de FN}\big(\theta_k,\ u_{0,k}\big) && \text{(integrada con un solver de EDO)}\\
V_k(t),\,R_k(t) &\sim \mathcal N\!\big(u_k(t),\ \sigma^2\big) && \text{verosimilitud (datos del ciclo)}
\end{aligned}
$$

con prior $m_0 = (0.2,\,0.3,\,\log 3,\,0)$ y $\Sigma_0=\operatorname{diag}(0.5,0.5,0.6,0.5)^2$
(débilmente informativo, centrado en los valores típicos de FN).

**La clave — qué hace $\tau$:**
- $\tau \to 0$: todos los ciclos comparten el mismo $\theta$ (**complete pooling**).
- $\tau \to \infty$: cada ciclo se ajusta solo (**no pooling**).
- $\tau$ intermedio: **shrinkage** — cada $\theta_k$ se encoge hacia $\mu$ tanto como la
  evidencia lo permite (óptimo sesgo-varianza, James-Stein). Y $\tau$ **mide la
  no-estacionariedad** entre crisis, con intervalo creíble.

**Inferencia:** NUTS/HMC sobre $(\mu, \tau, \sigma, \{\theta_k\})$. La posterior se
resume en $\hat\mu = \mathbb E[\mu]$, $\hat\tau = \mathbb E[\tau]$,
$\hat\sigma = \mathbb E[\sigma]$ — **el pool**.

---

## 4. Predicción de una crisis *held-out* (p. ej. 2008)

### 4a. Solo con el prior del pool (insuficiente)

Tomar $\theta \sim \mathcal N(\hat\mu, \operatorname{diag}(\hat\tau^2))$ e integrar FN
desde $u_0$ del ciclo. Da un ciclo **genérico**: reconstruye la inversión pero **no la
amplitud específica de los profits** (mediana plana, bandas enormes). No es el predictor.

### 4b. **Condicionada, h-step** (el predictor ganador)

Para cada **origen móvil** $\tau$ dentro del ciclo de la crisis:

1. **Condicionar** $\theta$ a los datos del ciclo **hasta $\tau$**, usando el pool como
   prior. Estimador MAP (= mínimo del negativo log-posterior, ambos Gaussianos):

$$
\hat\theta(\tau) \;=\; \arg\min_{\theta}\;
\underbrace{\frac{1}{2\hat\sigma^2}\!\!\sum_{t_i \le \tau}\!\big\|u(t_i;\theta,u_0) - (V_{t_i},R_{t_i})\big\|^2}_{\text{ajuste a los datos } \le\,\tau}
\;+\;
\underbrace{\sum_{j=1}^{4}\frac{(\theta_j - \hat\mu_j)^2}{2\,\hat\tau_j^{2}}}_{\text{prior del pool (regularización)}}
$$

2. **Pronosticar**: anclar en el **estado observado en $\tau$**,
   $u_\tau = (V_\tau, R_\tau)$, e integrar FN con $\hat\theta(\tau)$ hacia adelante $h$
   trimestres:

$$
\hat u(\tau{+}h) = \text{solución de FN}\big(\hat\theta(\tau),\ u_\tau\big)\Big|_{\,h},
\qquad
e_h(\tau) = \hat u(\tau{+}h) - u_{\tau+h}.
$$

3. **Rodar** $\tau$ y acumular el error por horizonte:
   $\text{RMSE}(h) = \sqrt{\operatorname{mean}_\tau\, e_h(\tau)^2}$ (en unidades YoY).

Esto es **directamente comparable al VAR**: ambos ajustan con historia $\le\tau$ y
pronostican $h$ pasos. Horizontes $h \in \{1,2,4\}$ trimestres (headline $h{=}4$),
**iterated** (se integra la propia dinámica).

---

## 5. La ablación "pool vs sin-pool"

El segundo término (prior del pool) es lo que transfiere información **de las otras
crisis**:

$$
\hat\theta_{\text{sin pool}}(\tau) = \arg\min_{\theta}\;
\frac{1}{2\hat\sigma^2}\!\!\sum_{t_i \le \tau}\!\big\|u(t_i;\theta,u_0) - (V_{t_i},R_{t_i})\big\|^2
\quad(\text{sin el término de prior}).
$$

Comparar RMSE con vs sin el término de prior **mide si "cada crisis aporta a 2008"**.
Empíricamente: el pool **parte el error casi al medio a horizonte corto** (h1 profits
2008: ~13.5 con pool vs ~23 sin) — la evidencia central del trabajo.

---

## 6. Por qué este modelo gana

| Propiedad | Mecanismo matemático | Efecto |
|---|---|---|
| Amplitud y forma del ciclo | ciclo límite + relajación de FN | captura profits y la asimetría (LV no podía) |
| Pronóstico a $h$ corto | integrar la dinámica no lineal desde $u_\tau$ | le gana a VAR/AR en h1–h2 |
| Pocos datos sin overfit | shrinkage del prior del pool $\tfrac{(\theta-\hat\mu)^2}{2\hat\tau^2}$ | regulariza; reduce varianza |
| "Cada crisis aporta" | el prior $\hat\mu,\hat\tau$ viene de las otras crisis | transfiere información a 2008 |
| Incertidumbre honesta | posterior completo $(\mu,\tau,\sigma)$ | bandas creíbles; $\tau$ = no-estacionariedad |

**Límite conocido:** a $h{=}4$ la extrapolación no lineal **deriva** y el error sube
por encima de los baselines — el modelo sirve para **horizonte corto**, no largo.

---

### Referencias de implementación
- `src/experiment/fn_hierarchical.jl` — modelo §3, predicción §4a.
- `src/experiment/fn_forecast.jl` — predicción condicionada §4b, ablación §5.
- Diseño completo del experimento: `EXPERIMENT.md` (raíz).
