# La cadena de acumulación, explicada despacio

Nota intuitiva del modelo físico del paper (sección "El modelo físico").
Objetivo: entender qué es la "cadena de $N$ etapas" y por qué termina dando un
núcleo Gamma, sin fórmulas más allá de lo necesario.

---

## 0. Qué estamos tratando de capturar

La inversión de hoy **no** afecta la ganancia de hoy ni la de un trimestre fijo
después. La plata se invierte, se construye una planta, tarda en operar, recién
ahí agrega capacidad, y esa capacidad recién presiona la rentabilidad cuando
produce más de lo que el mercado absorbe. Todo eso **lleva tiempo y, sobre todo,
llega repartido**: un poco a los 3 trimestres, más a los 5, algo a los 7…

Necesitamos un objeto matemático que convierta "inversión de golpe" en
"presión sobre la ganancia, demorada y desparramada en el tiempo". Esa es la
cadena.

---

## 1. Una sola etapa: "perseguir con constante de tiempo $\theta$"

La ecuación de una etapa es:

$$\frac{dm_j}{dt}=\frac{m_{j-1}-m_j}{\theta}$$

Léela así: **$m_j$ se mueve para alcanzar a $m_{j-1}$, y la velocidad con que se
mueve es proporcional a cuán lejos está.**

- $m_j$ = dónde estoy (mi **salida**).
- $m_{j-1}$ = a dónde quiero llegar = mi **entrada**: lo que me llega de la etapa
  anterior y que persigo. (Las etapas van en fila —como un teléfono descompuesto—;
  cada una solo "escucha" a la que tiene justo antes. La primera escucha
  directamente a la inversión: $m_0 = I$.)
- $(m_{j-1}-m_j)$ = el gap, lo que me falta.
- $\theta$ = mi "lentitud": cuánto tardo en cerrar el gap.

### Analogía: el café que se enfría

$m_j$ = temperatura del café. $m_{j-1}$ = temperatura del ambiente.
El café se acerca al ambiente **rápido cuando hay mucha diferencia y cada vez más
lento a medida que se acerca**. Nunca pega un salto. Eso es exactamente la
ecuación.

### Con números ($\theta = 2$ trimestres)

Supongamos que la entrada salta a 10 y $m_j$ arranca en 0:

| tiempo        | $m_j$ | gap que queda |
|---------------|-------|---------------|
| 0             | 0,0   | 100 %         |
| $\theta$ (2)  | 6,3   | 37 %          |
| $2\theta$ (4) | 8,6   | 14 %          |
| $3\theta$ (6) | 9,5   | 5 %           |

Regla: **cada $\theta$ trimestres se cierra ~63 % del gap que quedaba.**

- $\theta$ chico → persigue rápido, casi sin demora.
- $\theta$ grande → persigue lento, queda rezagado y suavizado.

**Una etapa = un retardo suave.** La salida es una copia *demorada y alisada* de
la entrada.

### ¿Por qué "persigue" y no solo "alcanza"?

En el café el ambiente estaba **quieto**, así que $m_j$ lo *alcanzaba*. Pero en la
cadena real $m_{j-1}$ **se mueve todo el tiempo** (es la salida de la etapa
anterior, que también cambia). Entonces $m_j$ corre detrás de un **blanco móvil**
y **nunca lo alcanza del todo: queda siempre un poco atrás.** Esa distancia que
queda atrás **es el retardo**.

Ejemplo: si $m_{j-1}$ sube de a 1 por trimestre ($0,1,2,3,\dots$) y $\theta=2$,
$m_j$ lo persigue pero queda rezagado un gap ~constante:

| trim | $m_{j-1}$ (persigo) | $m_j$ (persiguiendo) |
|------|---------------------|----------------------|
| 0    | 0                   | 0                    |
| 4    | 4                   | ~2,3                 |
| 8    | 8                   | ~6,2                 |

Analogía: un corredor persiguiendo a otro. Si el de adelante **para**, lo
alcanzás (café). Si el de adelante **sigue corriendo**, te quedás siempre unos
metros atrás — y esos metros son la demora.

---

## 2. Un solo "golpe" de inversión a través de una etapa

Si en vez de un escalón metemos un **pulso** (invertí fuerte un trimestre y
listo), una sola etapa responde con una **caída exponencial**: salta y se va
apagando como $e^{-t/\theta}$.

```
una etapa:   █
             █▇▅▃▂▁          (pegado al origen, solo baja)
```

Problema: esto pone el efecto **máximo en el instante 0**. Pero nosotros sabemos
que la inversión tarda; el efecto no debería ser máximo al toque.

---

## 3. Encadenar etapas: sube, hace pico, baja

Ahora ponemos varias etapas **una atrás de otra**: la inversión entra en la
etapa 0, su salida alimenta la 1, la de la 1 alimenta la 2, etc. Cada una demora
y alisa un poco más.

El resultado de un pulso ya **no** está pegado al origen. Pasa a:

```
1 etapa:    █▇▅▃▂▁
3 etapas:   ▁▃▅▆▅▃▂▁          ← sube, hace PICO, baja
8 etapas:     ▁▂▄▆█▆▄▂▁       ← pico más definido y centrado
```

Intuición: el "golpe" de inversión tiene que **atravesar todas las etapas** antes
de pegar fuerte. Las primeras lo demoran, así que el efecto máximo aparece *más
tarde*, no en t=0. Cuantas más etapas, más definido y centrado el pico.

Esto es lo que queríamos: **efecto demorado y repartido**, con un máximo en el
medio.

---

## 4. Eso es exactamente un núcleo Gamma

El "truco de la cadena lineal" (resultado estándar) dice: encadenar $N$ etapas
iguales de constante $\theta$ da **exactamente** una curva Gamma (Erlang):

$$w(t)\;\propto\;t^{\,N-1}\,e^{-t/\theta}$$

No hace falta entender la fórmula. La lectura en castellano es probabilística:

- El tiempo que una unidad de inversión tarda en **cruzar una etapa** es al azar,
  con media $\theta$.
- El tiempo **total** hasta salir de la cadena es la suma de $N$ de esas demoras.
- Sumar $N$ demoras independientes ⇒ **las medias se suman y las varianzas se
  suman**. Por eso:
  - retardo típico (centro): $\mu = N\theta$
  - dispersión (ancho): $\sqrt{N}\,\theta$

| | efecto |
|---|---|
| **muchas etapas** ($N$ grande) | pico angosto, concentrado alrededor de $\mu$ |
| **pocas etapas** ($N$ chico)   | pico ancho y asimétrico |

Ese $w(t)$ son los **pesos** $w_j$: "qué fracción de la inversión de hace $j$
trimestres recién ahora termina de pesar sobre la rentabilidad". La **inversión
acumulada** $m$ es el promedio ponderado de la inversión pasada con esos pesos.

---

## 5. Qué de todo esto estimamos (y qué no)

Acá está la sutileza del paper:

- De los dos parámetros $N$ y $\theta$, **solo se estima su producto**
  $\mu = N\theta$ — el **centro** del retardo (~5 trimestres, ~1 año).
- La **forma** (el ancho, que depende de $N$) **no se identifica**: dos cadenas
  muy distintas (una casi puntual, una ancha) ajustan los datos casi igual. Así
  que $N$ se **fija** por supuesto (en el código, $N=8$), no se recupera.

> En una línea: la cadena es la **explicación física** de por qué el efecto de la
> inversión es un retardo distribuido con forma de Gamma. Pero de toda esa
> estructura, lo único que los datos dejan medir es **dónde está el centro**
> ($\mu$), no qué tan ancha es la campana.

---

## Resumen de una frase

> Cada etapa "persigue" a la anterior (la copia con demora y suavizado); al
> encadenar varias, un golpe de inversión sale convertido en una campana
> (Gamma) que **sube, hace pico a ~1 año y baja** — y eso es la presión demorada
> de la inversión sobre la ganancia. De esa campana solo medimos el **centro**.
