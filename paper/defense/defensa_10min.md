# Defensa — 10 minutos

**Tesis:** *Un retardo distribuido de acumulación en el ciclo ganancias–inversión: identificación de un feedback débil y sus límites.* — Juan I. Tollo

Objetivo de tono: **honesto y mecanicista**. El resultado vendible no es "encontré un ciclo", es **dónde cae la frontera de identificabilidad**: qué fija una señal débil y qué no.

---

## Guion minuto a minuto (~1.300 palabras ≈ 10 min)

### 0:00–1:15 · La pregunta (engancha rápido)
> Las ganancias y la inversión forman un ciclo. Ganancias altas estimulan inversión; esa inversión, al acumularse en capacidad instalada, erosiona las ganancias después. La tradición —Tapia 2023, Astarita 2012— lo lee como **sobreacumulación** y lo formaliza como un sistema **depredador–presa**.
>
> Mi pregunta es acotada y mecánica: **¿ese feedback de la inversión hacia las ganancias es instantáneo, como imponen esos osciladores, o es retardado, con un tiempo característico que refleja la maduración de la inversión?** La respuesta de los datos es: **retardado, alrededor de un año.**
>
> Y el aporte real no es el fenómeno —es previo—, sino **el modelo y el análisis de identificabilidad**: con una señal que explica apenas el 3% de la varianza, ¿qué se puede fijar y qué no?

### 1:15–2:30 · Datos y la decisión metodológica clave
> Series trimestrales de EE.UU., 1948–2026, FRED: ganancias corporativas e inversión privada bruta. 312 trimestres en tasas de crecimiento.
>
> Trabajo con **crecimiento log trimestre a trimestre**, y esto es una decisión, no un detalle. La transformación interanual habitual —YoY— **no sirve**: es una diferencia a 4 trimestres, justo la escala del feedback que quiero medir, así que el filtro se come la señal y el lag estimado se corre a 2 trimestres. Con la diferencia trimestral el lag de ~1 año aparece estable.

### 2:30–4:15 · El hallazgo empírico (FIGURA CCF — la principal)
> *(Mostrar Fig. CCF, `p7b_ccf_surrogate.pdf`.)*
>
> La herramienta es la **correlación cruzada** ganancias↔inversión. Y decide algo simple: si la función **oscila** —cruza a negativo en los rezagos— hay ciclo; si **decae a cero**, es co-movimiento puro.
>
> Veo **dos cosas a la vez**. Primero, un pico contemporáneo fuerte, ρ(0)≈+0,5: las dos series se mueven juntas y se desploman juntas en cada recesión. La ganancia adelanta a la inversión por un trimestre —es el **acelerador**, la dirección rápida y dominante.
>
> Segundo, y es lo que estudio: un **valle negativo en el rezago −4**, ≈−0,2. La inversión de hace un año se asocia con **menos** ganancia hoy. Ese es el feedback de sobreacumulación: débil, no visible a ojo, pero ahí.
>
> Esta combinación **descarta los osciladores de fase fija** —Lotka–Volterra, FitzHugh–Nagumo—. En ellos el desfase es fijo, un cuarto de ciclo por construcción, lo que **fuerza ρ(0)=0**. Los datos muestran exactamente lo contrario: ρ(0)≈+0,5. Así que el ciclo no puede venir de esa familia de modelos.

### 4:15–6:00 · El modelo físico (la idea elegante)
> Si el feedback es retardado, necesito un modelo donde **el retardo lo fijen los datos, no la forma del modelo**. La clave: el retardo no es un número fijo. La inversión se construye, entra en operación, agrega capacidad, y recién cuando esa capacidad produce más de lo que el mercado absorbe presiona la rentabilidad. Sus efectos llegan **repartidos en el tiempo**.
>
> Lo modelo como una **cadena de N etapas**, cada una un suavizado exponencial que persigue a la anterior con constante de tiempo θ. El **truco de la cadena lineal** (Smith 2011) dice que la respuesta de esa cadena es **exactamente un núcleo Gamma (Erlang)**, `w(t) ∝ t^(N−1) e^(−t/θ)`.
>
> La lectura probabilística lo hace transparente: el tiempo total que tarda una unidad de inversión en atravesar la cadena es la suma de N demoras exponenciales. Medias y varianzas se suman: el retardo típico es **μ = Nθ** y su dispersión **√N·θ**.
>
> El sistema agregado es **lineal**, y a diferencia de Lotka–Volterra —donde el ciclo nace del producto de las variables y el desfase queda fijado—, **acá el ciclo lo genera el retardo, no una no linealidad**. El término clave es `+k·m` con k<0: la inversión madurada hace ~μ trimestres deprime la ganancia de hoy.

### 6:00–7:45 · Estimación e identificabilidad (el corazón)
> Estimo μ con un **inverso bayesiano** del modelo físico —Turing.jl, NUTS—. En vez de un número, devuelve un **posterior** con incertidumbre. Y lo coteo con la CCF, que no supone el modelo.
>
> *(Mostrar Fig. bayes, `p17_bayes_lag.pdf`.)*
>
> Tres resultados:
> 1. **El centro del retardo se identifica:** μ ≈ 4,9 trimestres, IC [3,0; 7,3], mucho más angosto que el prior. El dato informa μ. Consistente con el valle de la CCF y con el R² del modelo físico, máximo en μ≈5.
> 2. **El signo del efecto se identifica:** k ≈ −0,40, IC95% [−0,63; −0,12], **no cruza cero**. La inversión acumulada deprime la ganancia posterior. Capital que al acumularse erosiona su propia rentabilidad.
> 3. **Lo que NO se identifica:** el **ancho** del núcleo. El perfil de verosimilitud en la dispersión es **plano** —dos núcleos, uno casi puntual y uno ancho, ajustan igual—. Los datos no distinguen un retardo distribuido de uno puntual.

### 7:45–8:45 · Por qué creerle (controles) y los límites
> Cuatro controles de identificabilidad. El que más importa: un **null por surrogates AR** —2.000 pares de series que conservan la autocorrelación y el co-movimiento contemporáneo pero rompen el feedback cruzado retardado—. El valle observado a −4 cae **fuera** de esa banda, p≈0,006. No es un artefacto de autocorrelación. Y un **placebo**: el valle aparece con la misma fuerza fuera de las crisis (p≈0,35), así que es **estructural, no firma de las crisis**.
>
> Dos límites honestos: con 3% de varianza y solo ~9 crisis utilizables, los datos **no separan** un retardo distribuido de uno puntual, ni el mecanismo de un **VAR(p≥4)** lineal de memoria larga —en muestra reproduce el mismo valle—.

### 8:45–10:00 · Conclusión y el aporte transferible
> El cuadro es consistente y honesto. **Existe** un feedback negativo rezagado, consistente con sobreacumulación, con retardo de ~1 año. Lo que importa de ese feedback es **su signo, no su magnitud**: una presión persistente y estructural sobre la rentabilidad, débil en varianza pero consistente en su dirección.
>
> El aporte metodológico, en una línea: **bajo señal débil, recuperar un parámetro estructural es cuestión de cómo se estima, no de cuántos datos hay.** Una calibración bayesiana directa fija el centro del retardo donde una red flexible (PINN) lo disuelve y un VAR en forma reducida no lo separa de sus rezagos.
>
> Y el resultado transferible es **dónde cae la frontera**: qué fija la señal y qué no. Cruzarla no requiere más método —requiere datos nuevos: tiempos de construcción sectoriales que predigan el núcleo macro, un test que un VAR en forma reducida no puede pasar. **Ahí, y no en el ajuste en-muestra, el mecanismo sería refutable.**

---

## Las 3 frases-ancla (si te quedás en blanco, decí estas)
1. "La CCF oscila en vez de decaer: hay co-movimiento contemporáneo **y** un valle negativo retardado. Eso descarta los osciladores de fase fija."
2. "Lo que la señal débil fija: el **centro** del retardo (~1 año) y el **signo** del efecto (k<0). Lo que no fija: el **ancho** del retardo y su **separación de un VAR**."
3. "Mi aporte no es el fenómeno —es previo—, es el modelo identificable y dónde cae la frontera de lo que estos datos permiten."

---

## Preguntas probables del jurado + respuestas

**P1. Si el VAR lineal reproduce el mismo valle, ¿qué agrega tu modelo?**
> En muestra, nada en ajuste —y lo digo explícitamente—. Mi modelo es una **reparametrización interpretable** de ese contenido lineal. La diferencia es fuera de muestra: el modelo estructural ata μ a una cantidad **medible aparte** —tiempos de construcción y maduración sectoriales—. Esa es una predicción que un VAR en forma reducida no puede hacer; ahí el mecanismo es refutable. El VAR ajusta, pero no compromete nada.

**P2. ¿No es sobreajuste? ¿O p-hacking con la transformación?**
> Seis parámetros para 312 datos, no hay sobreajuste. La transformación log-trimestral no la elegí por el resultado: la justifico mecánicamente —la YoY mezcla un año de datos, la misma escala del feedback—, y muestro que con YoY el lag se corre a 2 trimestres. Es una decisión de diseño, documentada y con su razón.

**P3. ¿Por qué Gamma/Erlang y no otra forma de retardo?**
> No la impongo por gusto: sale del **truco de la cadena lineal** (Smith 2011). Si modelás la maduración como una cadena de etapas exponenciales —que es la historia económica: construcción, entrada en operación, saturación—, la respuesta **es** exactamente un núcleo Gamma. Y de todos modos el dato no identifica el ancho, así que no estoy apostando a una forma fina: solo uso el centro, μ, que sí se identifica.

**P4. 3% de varianza es nada. ¿Por qué te importa?**
> Porque **significativo ≠ grande**, y **débil en varianza ≠ irrelevante en dirección**. El IC95% de k no cruza cero: el signo es consistente y estructural. Una presión de sobreacumulación no necesita explicar la varianza del ciclo para ser real; su dirección persistente es el punto sustantivo. La varianza la domina el acelerador, que no es lo que estudio.

**P5. ¿Sobreacumulación o profit-squeeze? ¿Cómo los distinguís?**
> Honestamente, **con estas dos series no los distingo**, y lo digo en limitaciones. Una caída de ganancia tras inversión es compatible con ambos: sobreacumulación (capacidad que el mercado no valida) o profit-squeeze (la actividad sube costos y comprime la participación). Distinguir el canal exige datos de ventas, inventarios y utilización, ausentes acá. Mi afirmación se queda en el feedback retardado negativo, que es lo que los datos sí sostienen.

**P6. ¿Por qué bayesiano y no el PINN, que está de moda?**
> Probé el PINN y **no identifica** el retardo —está en el apéndice—. La red es flexible: ajusta los datos satisfaciendo la ODE para un rango ancho de μ, así que el costo queda plano en μ y la señal débil que debería anclarlo se **absorbe** en la red. El bayesiano —seis parámetros directos, sin red— deja que ese término débil informe μ. Es justamente la lección: con señal débil, la flexibilidad de la red **destruye** la identificabilidad que la calibración directa preserva.

**P7. Solo EE.UU. ¿No deberías sumar más países?**
> Pensé esa vía y la descarto razonadamente. EE.UU. es prácticamente la única serie larga y confiable, y sus crisis están espaciadas ~una por década, así que acumular historia rinde poco —la señal es genuinamente escasa—. La vía realista no es más países sino **más resolución dentro del mismo sistema**: tiempos de construcción sectoriales o datos de mayor frecuencia.

**P8. ¿Por qué excluís 2008 y 2020?**
> Para mostrar que el resultado **no depende de las crisis extremas**. El valle y el lag son robustos a excluirlas —lo verifico—. Si el efecto solo apareciera con 2008/2020, sería una firma de outliers, no un patrón estructural. Que sobreviva sin ellas es la evidencia de que es estructural.

**P9. ¿Qué harías distinto / trabajo futuro?**
> El límite conceptual abierto: mi modelo lineal captura la acumulación y su reflujo **suave**, pero no la **destrucción/desvalorización de capital** —la descarga abrupta y no lineal de la sobreacumulación en la crisis—. Captarla pide extender la cadena con un mecanismo de crisis no lineal o de cambio de régimen. Pero anticipo que, con 3% de señal y nueve episodios, esa mejora **podría no ser identificable** —el mismo límite que estructura toda la tesis—. Acoplar un ciclo por retardo con una resolución de crisis no lineal es, hasta donde sé, un problema abierto.

**P0. (Casi seguro) Tenés 6 parámetros y solo discutís k. ¿Y los signos del resto? ¿Tienen sentido?**
> Los signos son las dos flechas del ciclo escritas en ecuaciones. **a > 0** (ganancia → inversión) es el acelerador; **c > 0** (inversión → ganancia contemporánea) es el efecto Kalecki —el gasto de inversión genera ganancia en el mismo período—. Esos dos positivos forman el lazo rápido que produce el co-movimiento contemporáneo ρ(0)≈+0,5. **b > 0** y **d > 0** son amortiguación —mantienen el sistema estacionario, no explota—. Y **k < 0** es la sobreacumulación, el feedback retardado que estudio, el que da el valle en −4. Sobre identificación: a y c quedan bien fijados porque la señal contemporánea fuerte (~24% de varianza) los ancla; el canal difícil es el débil, y ahí mi afirmación es acotada —la señal alcanza para μ y el signo de k, no para lo fino—.

**P10. (Técnica) ¿Cómo sabés que μ está identificado y no es el prior?**
> El posterior de μ es **mucho más angosto que el prior** —se ve en la figura—. Si el dato no informara, el posterior reproduciría el prior. Además converge con dos métodos independientes: el valle de la CCF (que no supone el modelo) y el máximo de R² del modelo físico, ambos en μ≈5. μ es la **media** del núcleo, integra todo el pozo de la CCF, por eso cae un poco más allá del rezago más profundo.

---

## Checklist pre-defensa
- [ ] 3 figuras a mano: CCF (`p7b`), modelo físico/identificabilidad (`p11`), bayes (`p17`).
- [ ] Si solo podés mostrar una: **la CCF**. Es el hallazgo.
- [ ] Tener los 3 números en la punta de la lengua: **μ≈4,9 [3,0;7,3]**, **k≈−0,40 [−0,63;−0,12]**, **3% vs 24% de varianza**.
- [ ] No oversell. La fuerza de la defensa **es** la honestidad sobre los límites. El jurado va a respetar eso más que un resultado inflado.
