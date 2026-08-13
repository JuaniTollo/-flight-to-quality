# El cálculo del posterior en `p17_bayes_lag.jl`

Apunte de estudio para la defensa: cómo se calcula (en realidad, cómo se *muestrea*)
el posterior del inverso bayesiano del retardo.

## 1. La fórmula exacta

Sea $\theta = (\mu, c, e, k, \beta_0, \sigma)$ el vector completo de parámetros
(son 6, no solo $\mu$). El teorema de Bayes:

$$
p(\theta \mid P, I) = \frac{p(P \mid \theta, I)\; p(\theta)}{p(P \mid I)}
$$

- $p(\theta)$ — **prior**: el producto de las densidades declaradas en el modelo
  (`μ ~ truncated(Normal(4,3), 0.5, 12)`, `k ~ Normal(0,1)`, etc.).
- $p(P \mid \theta, I)$ — **verosimilitud**: qué tan probable era observar la serie
  real de ganancias si los parámetros valieran $\theta$.
- $p(P \mid I)$ — **evidencia**: constante que no depende de $\theta$.
  Clave: para comparar valores de $\theta$ entre sí, el denominador **no importa**.
  Siempre se trabaja con el posterior *no normalizado*:
  $p^*(\theta) = \text{verosimilitud} \times \text{prior}$.

## 2. Cómo se evalúa la verosimilitud

Es lo que hace el loop `for t in (K+2):n` cada vez que Turing "visita" un $\theta$
candidato. Fijado un $\theta$ concreto (p. ej. $\mu=5$, $c=0.3$, $e=0.4$, $k=-0.4$,
$\beta_0=0$, $\sigma=0.8$):

1. Con $\mu=5$ se calculan los pesos del núcleo Gamma: `w = gamma_w(5.0, 10)`
   (once números que suman 1).
2. Para cada trimestre $t$ se arma la predicción:
   $\text{pred}_t = \beta_0 + c\,I_t + e\,P_{t-1} + k\sum_j w_j I_{t-j}$.
3. El residuo $P_t - \text{pred}_t$ tiene densidad bajo $N(0,\sigma)$:
   alta si el residuo es chico, baja si es grande.
4. La verosimilitud total es el **producto** de esas densidades sobre todos los $t$
   (en la práctica, la **suma de logaritmos**, por precisión numérica).

La log-verosimilitud es, salvo constantes:

$$
\log p(P \mid \theta, I) = -\sum_t \frac{(P_t - \text{pred}_t)^2}{2\sigma^2} - n\log\sigma
$$

El primer término es (menos) la suma de cuadrados de residuos: **por eso el máximo
del posterior con priors chatos coincide con mínimos cuadrados**. El bayesiano no es
magia distinta del OLS; es el mismo criterio de ajuste, más los priors, y explorado
*entero* en vez de solo en su máximo.

El log-posterior no normalizado es una sola función escalar:

$$
\log p^*(\theta) =
\underbrace{-\sum_t \frac{(P_t - \text{pred}_t(\theta))^2}{2\sigma^2} - n\log\sigma}_{\text{ajuste a los datos}}
\;+\;
\underbrace{\log p(\mu) + \log p(c) + \log p(e) + \log p(k) + \log p(\beta_0) + \log p(\sigma)}_{\text{priors}}
$$

Turing sabe evaluar ese número (y su gradiente, por diferenciación automática)
en cualquier $\theta$.

## 3. Por qué no hay fórmula cerrada

Si $\mu$ fuera conocido, el modelo sería lineal en $(c, e, k, \beta_0)$ y el
posterior tendría forma analítica (regresión bayesiana clásica). Pero $\mu$ entra
**adentro** de la exponencial del núcleo Gamma — no linealmente — así que la
integral que normaliza el posterior no se puede hacer a mano. Además, para reportar
"el posterior de $\mu$ solo" habría que marginalizar sobre las otras 5 dimensiones:

$$
p(\mu \mid P, I) = \int p(\mu, c, e, k, \beta_0, \sigma \mid P, I)\;
dc\, de\, dk\, d\beta_0\, d\sigma
$$

Una integral en 5 dimensiones sin forma cerrada. Ahí entra el muestreo.

## 4. La idea de MCMC: reemplazar integrar por muestrear

El truco central: si conseguís puntos $\theta^{(1)}, \theta^{(2)}, \dots,
\theta^{(1000)}$ **distribuidos según el posterior**, toda cantidad de interés es un
promedio sobre esos puntos:

- media posterior de $\mu$ ≈ promedio de los $\mu$ muestreados;
- IC 95% ≈ percentiles 2,5 y 97,5 de los $\mu$ muestreados (la función `q` del script);
- la marginalización en 5 dimensiones sale **gratis**: para el posterior de $\mu$
  solo, se ignoran las otras coordenadas de cada muestra. Eso hace
  `vec(Array(chain[:μ]))` — tira las otras columnas y listo.

MCMC construye esos puntos como una caminata: desde el $\theta$ actual se propone un
$\theta$ nuevo, y se acepta o rechaza comparando $p^*(\theta_{\text{nuevo}})$ con
$p^*(\theta_{\text{actual}})$ — solo cocientes, por eso el denominador de Bayes
nunca hace falta. Con la regla de aceptación correcta (Metropolis–Hastings), la
caminata pasa en cada región un tiempo proporcional a su probabilidad posterior:
donde el posterior es alto se acumulan muestras; donde es bajo, casi no pisa.

## 5. Qué agrega NUTS

Metropolis "a ciegas" (saltos aleatorios) es lentísimo en 6 dimensiones: casi todas
las propuestas caen en zonas de baja probabilidad y se rechazan.

**HMC (Hamiltonian Monte Carlo)** usa el gradiente del log-posterior para proponer
inteligente: trata $-\log p^*(\theta)$ como una superficie de energía, le da a la
"partícula" un impulso aleatorio y la deja deslizarse siguiendo la física. Por eso
el núcleo Gamma tenía que ser **diferenciable en $\mu$** — sin gradiente no hay HMC.
Eso produce propuestas lejanas que igual se aceptan casi siempre.

**NUTS (No-U-Turn Sampler)** es HMC que decide solo cuánto dura cada trayectoria: la
corta cuando detecta que empezó a dar la vuelta en U (volver sobre sí misma), y así
no hay que calibrar la longitud a mano. El `0.65` de `NUTS(0.65)` es la tasa de
aceptación objetivo con la que auto-ajusta el tamaño de paso durante el *warm-up*
(muestras iniciales que se descartan y no están en las 1000 finales).

## 6. El circuito completo en el script

Cada una de las 1000 iteraciones hace:

1. NUTS propone un $\theta$ nuevo usando gradientes.
2. Turing evalúa el log-posterior (loop del modelo: pesos Gamma → predicciones →
   residuos → priors).
3. Acepta/rechaza y guarda $\theta$ en la cadena.

Al final queda una matriz de 1000×6; el histograma de la columna $\mu$ es el
posterior marginal, y compararlo contra muestras del prior es el test de
identificabilidad: si el dato no informara, ambos histogramas se superpondrían.

## Resultado (results/p17_bayes_lag.csv)

| parámetro | media | IC 95% |
|---|---|---|
| $\mu$ (retardo, trim.) | 4,88 | [3,05; 7,29] |
| $k$ (sobreacumulación) | −0,38 | [−0,63; −0,12] |

El posterior de $\mu$ es mucho más angosto que el prior ([0,5; 12]) y $k$ no cruza
cero ⇒ el retardo (~5 trimestres ≈ 1 año) y el efecto de sobreacumulación **están
identificados por el dato**, no impuestos por el prior.

## La intuición en una frase

**El posterior no se "calcula", se "visita"**: la cadena es un explorador que
recorre el espacio de parámetros pasando más tiempo donde datos y priors están más
de acuerdo, y la foto de por dónde anduvo es la distribución que se reporta.

## ¿Por qué una sola ecuación y no el sistema de dos?

El modelo estructural del trabajo es el sistema **lineal con retardo distribuido**
(NO un depredador-presa: LV/FN de fase fija son la alternativa que la tesis descarta;
acá el ciclo nace del retardo, no de una no linealidad):

$$
\frac{dP}{dt} = c\,I - d\,P + k\,m, \qquad
\frac{dI}{dt} = a\,P - b\,I, \qquad
m(t) = \sum_j w_j(\mu)\, I(t-j)
$$

`p17` **no** invierte el sistema completo: invierte solo la **primera ecuación**,
discretizada, en forma reducida — condicionando en la inversión observada. La
regresión $P_t = \beta_0 + c\,I_t + e\,P_{t-1} + k\,m_t$ es la ecuación de $P$ en
tiempo discreto (el término $e\,P_{t-1}$ absorbe el decaimiento $-d\,P$). En el
modelo de Turing, `I` entra como dato y nunca aparece un `I[t] ~ ...`; solo hay
verosimilitud para P. Se calcula $p(\theta \mid P, I)$ vía $p(P \mid I, \theta)$.

Por qué es legítimo y conviene:

1. **La pregunta es sobre una sola flecha del sistema** (I→P: ¿hay efecto de
   sobreacumulación y con qué retardo?). La otra ecuación ($dI/dt = aP - bI$, cómo
   las ganancias empujan la inversión) no contiene a $\mu$; la relación P→I la
   documentan la CCF y el estudio de evento.
2. **Robustez**: un error de especificación en la ecuación de I no contamina la
   inferencia sobre $\mu$, porque se usa la I real, no la predicha.
3. **El costo (reconocerlo si lo objetan)**: condicionar en I asume que $\varepsilon_t$
   no retroalimenta a la inversión usada como regresor (exogeneidad débil). En un
   sistema con feedback es una aproximación — el precio de la forma reducida.
4. **División del trabajo en la tesis**: el inverso diferenciable del sistema
   completo (PINN con red sustituta) no mejora la identificación (ver Apéndice);
   p17 es el inverso bayesiano en forma reducida, con incertidumbre cuantificada.

Si se quisiera el bayesiano del sistema completo, bastaría agregar la segunda
ecuación como línea de verosimilitud en el loop (p. ej.
`I[t] ~ Normal(γ0 + a*P[t-1] + b*I[t-1], σ_I)`) y el posterior sería conjunto sobre
ambas. No está hecho porque no agrega información sobre $\mu$, que vive solo en la
ecuación de P.

## Posibles preguntas de defensa

- *¿Cómo sabés que la cadena convergió?* → diagnósticos: $\hat{R}$ ≈ 1, ESS alto,
  trace plots estacionarios (Turing los reporta en `describe(chain)`).
- *¿Por qué NUTS y no Metropolis?* → gradientes: eficiencia en 6 dimensiones.
- *¿El prior no está eligiendo el resultado?* → test posterior vs. prior: el
  posterior se concentra (desvío mucho menor); con prior débil el dato domina.
- *¿Qué relación tiene con el OLS/CCF?* → mismo criterio de ajuste en el máximo;
  el agregado es el intervalo de credibilidad y tratar el lag como continuo.
