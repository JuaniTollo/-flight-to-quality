# Optimización con restricciones y dualidad lagrangiana

> Notas sobre la formulación con restricciones del ajuste de EDOs (base para PINNs).
> Basado en `third_parties/DM2026-Curso/clases/clase8.md` (No8 - PINNs).

El punto central es que **un mismo problema se puede escribir de dos formas
equivalentes: sin restricciones o con restricciones**, y a veces conviene una u otra.

## El punto de partida: cuadrados mínimos "sin restricciones"

Tenés datos $y_i$ medidos en tiempos $t_i$, y un modelo $x(t,\theta)$ que depende de
parámetros $\theta$ (por ejemplo, los parámetros de una ecuación diferencial). Querés
elegir $\theta$ para que el modelo se parezca lo más posible a los datos:

$$\min_{\theta}\; \mathcal{L}(\theta,y)=\min_{\theta}\sum_{i=1}^{N}\big\|y_i - x(t_i,\theta)\big\|_2^2$$

Esto se ve como "sin restricciones" porque la única variable libre es $\theta$. **Pero
hay una trampa escondida:** para evaluar $x(t_i,\theta)$ tenés que *resolver la ecuación
diferencial* con esos $\theta$. O sea, $x$ ya viene "encadenada" a $\theta$ — no es libre.

## Reescribirlo "con restricciones"

La idea es **liberar a $x$** y tratarla como variable de optimización por derecho propio,
en vez de calcularla a partir de $\theta$. Pero si $x$ es libre, hay que *forzarla* a
comportarse como solución de la EDO. Eso se hace con una restricción:

$$\min_{\theta,\,x}\;\sum_{i=1}^{N}\big\|y_i - x(t_i)\big\|_2^2 \quad\text{sujeto a}\quad \begin{cases}\dfrac{dx}{dt}=f(x,t,\theta)\\[4pt] x(t_0)=x_0\end{cases}$$

Ahora optimizás sobre **dos cosas a la vez**: $\theta$ *y* la trayectoria $x$. La
restricción dice "la trayectoria que elijas tiene que satisfacer la EDO y arrancar en $x_0$".

## La equivalencia abstracta

En forma general, estás pasando de:

$$\min_{\theta}\; f(x(\theta)) \qquad\Longleftrightarrow\qquad \min_{\theta,\,x}\; f(x,\theta)\;\text{ sujeto a }\; G(x,\theta)=0$$

- **Izquierda (sin restricciones):** $x$ es una *función* de $\theta$. Vos ponés
  $\theta$, y "automáticamente" sale $x(\theta)$.
- **Derecha (con restricciones):** $x$ es variable independiente, y $G(x,\theta)=0$ es la
  condición que la ata a $\theta$.

La frase clave:

> *Si uno puede invertir $G(x,\theta)$ para obtener $x=x(\theta)$, volvemos al problema
> sin restricciones.*

Es decir: las dos formas son la misma cosa. Si supieras despejar $x$ en función de
$\theta$ desde la restricción $G=0$, sustituís y volvés al problema chico (solo en
$\theta$). El "invertir" puede ser **analítico** (despejar con álgebra) o **numérico**
(resolver la EDO con un solver).

## Qué es $G$ acá concretamente

La restricción $G(x,\theta)=0$ no es más que "ser solución de la EDO", escrita como algo
que vale cero:

$$G(x,\theta)=\begin{bmatrix}\dfrac{du}{dt}-f(u,t,\theta)\\[6pt] u(t_0)-u_0\end{bmatrix}=0$$

Dos componentes:

1. $\dfrac{du}{dt}-f(u,t,\theta)=0$ → la EDO se cumple en todo instante (el residuo de la
   ecuación es cero).
2. $u(t_0)-u_0=0$ → la condición inicial se cumple.

Cuando ambas valen cero, $u$ *es* la solución de la EDO. En el enfoque clásico, **esto se
hace con el solver numérico**: el solver es justamente quien "invierte $G$" — le das
$\theta$ y te devuelve la trayectoria $x(\theta)$ que satisface $G=0$.

## ¿Por qué molestarse con la versión con restricciones?

Esta es la motivación hacia PINNs / Lagrangiano:

- En el enfoque clásico, **resolver $G=0$ exactamente** (con el solver) en cada paso de
  optimización puede ser caro o difícil de diferenciar.
- La formulación con restricciones abre la puerta a **no resolver la EDO exactamente** en
  cada iteración, sino tratar la restricción con **dualidad lagrangiana**: agregás un
  término de penalización / multiplicadores de Lagrange y dejás que $x$ y $\theta$ se
  vayan ajustando *juntos*. Eso es esencialmente lo que hace una **PINN**: la red
  representa $x$, y el residuo de la EDO ($G$) entra como término de pérdida en vez de
  imponerse exactamente.
