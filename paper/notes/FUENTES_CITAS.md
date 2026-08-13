# Procedencia de las citas — DM2026 (flight_to_quality)

Última actualización: 2026-06-14.

Este archivo documenta, para cada cita del paper (`paper/main_es.tex`), **qué se
verificó, cómo y con qué evidencia**. La regla aplicada: si una afirmación teórica no
se pudo cotejar contra la fuente, **se sacó del texto** (no se deja "decorando").

Hay dos niveles:

- **VERIFICADA** — leímos la fuente (o la página exacta) y la cita refleja lo que dice.
- **NO VERIFICADA (solo título/secundaria)** — la entrada bibliográfica es correcta en
  título/año/revista, pero **no leímos el contenido**; se usa solo como antecedente de
  método o microfundación, sin atribuirle ninguna afirmación empírica que no podamos
  respaldar.

---

## 1. Tapia (2023) — VERIFICADA contra el libro (pp. 183–186)

**Cita bibliográfica:** Tapia, J.A. (2023). *Six Crises of the World Economy:
Globalization and Economic Turbulence from the 1970s to the COVID-19 Pandemic*.
Palgrave Macmillan / Springer Nature Switzerland AG, Cham.
ISBN 978-3-031-38734-0. DOI: https://doi.org/10.1007/978-3-031-38735-7

**Cómo se verificó:** se leyó el PDF del libro (capítulo empírico, sección de
ganancias e inversión). Offset PDF→página-de-libro = +16. Las páginas 183–186 contienen
el esquema causal endógeno y el modelo predador-presa que enmarca nuestro trabajo.

**Citas textuales (verbatim) que respaldan lo que afirmamos:**

- pp. 183–184 (pasaje continuo): *"I propose a causal endogenous scheme of the business
  cycle in which profits and investment are linked in a kind of predator–prey model.
  Movements in profits are followed some quarters later by movements in investment in the
  same direction, and movements in investment are followed by movements in profits in the
  opposite direction. Crises are preceded by drops in profitability."* (el salto de página
  183→184 cae dentro de la oración "Movements in profits …"; ganancias = presa, inversión =
  predador.)
- p. 186 (oración única, verbatim exacto): *"Statistical models also provide evidence—though
  in my analyses it looked as a weaker one—of another regularity between profits and
  investment, but this one in the direction from past investment to present profits, with
  the change in investment showing a negative lagged effect on the change in profits—so that
  a change in investment is generally followed by a change in profits in the opposite
  direction."*

**Qué autoriza afirmar en el paper:**
- La estructura **bidireccional** ganancias↔inversión (las dos direcciones).
- Que la dirección inversión→ganancias es **negativa y rezagada** ("algunos trimestres
  después").
- Que esa segunda dirección es la **más débil** (significativa en datos anuales, no en
  trimestrales). Esto coincide con nuestro propio EDA (β_m≈+0,02, t≈0,15, no signif.).
- El encuadre **predador-presa** con ganancias=presa, inversión=predador es de **Tapia
  (2023)**, no de Goodwin. Documentado además en el EDA histórico
  (`git show 16d7397:report/README.md`).

**Aparece en main_es.tex:** L52–59 (intro "La hipótesis económica"), L228 (Discusión).

---

## 2. Tapia Granados (2012) — RETIRADA del paper

Antes se citaba "Tapia Granados (2012)". Se **eliminó por completo** (3 menciones inline
+ bibitem) porque el abstract de ese artículo afirma una dirección de causalidad
(ganancias→inversión como predicción dominante) distinta de nuestra tesis principal
(efecto rezagado inversión→ganancias). No teníamos acceso al texto completo (Deep Blue
devolvió 403; Unpaywall: `is_oa: False`) para confirmar el matiz, así que se quitó en
lugar de citarla imprecisamente. El antecedente de Tapia que **sí** sostiene nuestro
encuadre es el libro de 2023 (entrada 1).

---

## 3. Astarita (2012) — VERIFICADA contra fuente primaria de Astarita

**Cita bibliográfica:** Astarita, R. (2012). "Crisis, sobrecapacidad y coyuntura."
*rolandoastarita.blog*, 22 de marzo de 2012.
https://rolandoastarita.blog/2012/03/22/crisis-sobrecapacidad-y-coyuntura/

**Por qué se cita a Astarita:** la lectura de las crisis como **sobreacumulación /
sobreinversión** (no subacumulación), donde la inversión que madura en capacidad instalada
presiona a la baja la tasa de rentabilidad, es la **contribución teórica propia de Astarita**
—la desarrolla en posts que ni siquiera mencionan a Tapia. Es exactamente el contenido
económico de nuestro término $-k\,m$ (sobreacumulación). El hallazgo empírico de que las
ganancias preceden a la inversión es de **Tapia**; el **mecanismo** de sobreacumulación es
de **Astarita**. Por eso van juntos (Astarita, 2012; Tapia, 2023).

**Citas textuales verificadas (fetch directo del post):**
- *"la sobrecapacidad... consiste en la sobreacumulación de capital fijo, esto es, en la
  construcción de plantas con una capacidad de producir una cantidad de mercancías muy
  superior a lo que puede absorber el mercado."*
- *"La sobrecapacidad ejerce una presión bajista sobre la tasa de rentabilidad..."*
- Encuadre "sobreacumulación, o sobreinversión, no por subacumulación" verificado además en
  Astarita, R. (2024) "Las crisis cíclicas y la LTDTG", rolandoastarita.blog, 1 nov 2024:
  *"no es una crisis por subacumulación o subinversión, sino por sobreacumulación, o
  sobreinversión."* (fuente secundaria, no citada en el paper para no multiplicar entradas.)

**Decisión editorial:** se cita **un solo** trabajo de Astarita (el de 2012, el más on-point
porque liga sobreacumulación de capital fijo → sobrecapacidad → presión bajista sobre la
rentabilidad, que es el mapeo exacto de $-k\,m$). El libro *El capitalismo roto* (2009) y el
coautoreado con Tapia (2011) existen pero no se pudieron consultar con número de página; no se
citan para no atribuir contenido sin verificar.

**Aparece en main_es.tex:** L57 (intro, atribución de la sobreacumulación), L228 (Discusión).

**Nota sobre el EDA:** el EDA histórico (`git show 16d7397:report/README.md`, L8, L168) atribuye la fase de sobreacumulación a
"Tapia draws from Astarita (Tapia 2023, p. 132)". La p. 132 del libro (nota 33) solo dice que
Astarita influyó en la lectura **general de Marx** de Tapia, no en el modelo predador-presa.
La atribución correcta del **mecanismo** de sobreacumulación es a obra propia de Astarita
(2012), como queda en el paper; conviene precisar esa línea del EDA si se reusa.

---

## 4. Goodwin (1967) — NO VERIFICADA (antecedente de MÉTODO, no de variables)

**Cita:** Goodwin, R.M. (1967). A growth cycle. En *Socialism, Capitalism and Economic
Growth*. Cambridge University Press.

**Estado:** entrada bibliográfica correcta (título/año/editorial estándar y conocidos).
**No releímos el texto de Goodwin** para este trabajo.

**Uso permitido en el paper:** únicamente como antecedente de la **dinámica de
Lotka–Volterra usada en macrodinámica**. El modelo de Goodwin es LV sobre
**empleo y participación salarial**, NO sobre ganancias e inversión. Por eso el texto
dice "usada en macrodinámica por Goodwin (1967)" de forma genérica y **NO** dice que
Goodwin analice ganancias. Esta precisión fue el motivo de la última edición.

**Aparece en:** L58.

---

## 5. Goldstein (1999) — NO VERIFICADA (solo título)

**Cita:** Goldstein, J.P. (1999). Predator–prey model estimates of the cyclical profit
squeeze. *Metroeconomica* 50(2), 139–173.

**Estado:** el **título** respalda lo que afirmamos (un modelo predador-presa estimado
para el ciclo de ganancias — "profit squeeze"). **No leímos el artículo completo.**

**Uso permitido:** como ejemplo de oscilador LV de **fase fija** estimado sobre el ciclo
de ganancias — que es el blanco de nuestra refutación de los 90° de desfase. No se le
atribuye ningún número ni resultado específico más allá de lo que el título sostiene.

**Aparece en:** L59.

---

## 6. Kalecki (1935) y Kydland–Prescott (1982) — NO VERIFICADAS (microfundación, título)

**Citas:**
- Kalecki, M. (1935). A macrodynamic theory of business cycles. *Econometrica* 3(3),
  327–344.
- Kydland, F.E., Prescott, E.C. (1982). Time to build and aggregate fluctuations.
  *Econometrica* 50(6), 1345–1370.

**Estado:** entradas bibliográficas correctas. **No releímos los textos.** El título de
Kydland–Prescott ("Time to build") respalda literalmente el concepto de **rezagos de
gestación / tiempo de construcción** de la inversión, que es para lo único que se las
cita (microfundación del retardo distribuido). No se les atribuye ningún resultado
empírico.

**Aparece en:** L235.

---

## 7. Referencias metodológicas (ML científico / lags) — estándar, no teóricas

Estas no son afirmaciones económicas; son referencias de método cuyas entradas
bibliográficas son estándar y correctas:

- Almon, S. (1965). *Econometrica* 33(1), 178–196. — rezagos distribuidos polinómicos.
- Raissi, Perdikaris, Karniadakis (2019). *J. Comp. Phys.* 378, 686–707. — PINNs.
- Rackauckas et al. (2020). arXiv:2001.04385. — universal differential equations.
- Ramsay, Hooker (2017). *Dynamic Data Analysis*. Springer. — gradient matching.

No se les atribuye ninguna afirmación sobre ganancias/inversión.

---

## Resumen de qué se editó por esta revisión

1. Reescrito el párrafo "La hipótesis económica" (L52–59): el encuadre predador-presa se
   ancla a **Tapia (2023, pp. 183–186)** con páginas verificadas; se dejó de insinuar que
   Goodwin analiza ganancias; se quitó la frase imprecisa sobre la tasa de ganancia (FROP)
   que no podíamos respaldar en Tapia.
2. Atribuida la idea de **sobreacumulación a Astarita (2012)** con fuente primaria propia de
   Astarita verificada (un solo trabajo), en la intro (L57) y la Discusión (L228).
3. Quitada por completo **Tapia Granados (2012)** — dirección causal del abstract distinta
   de nuestra tesis y sin acceso al texto.
4. Este archivo (`FUENTES_CITAS.md`) documenta la procedencia y el estado de cada cita.

## Poda de bibliografía (reducción a 5 páginas, 2026-06-14)

Para dejar solo lo discutido y citado en el texto, la bibliografía final del paper quedó en
**4 entradas**, todas efectivamente citadas:

- **Tapia (2023)** y **Astarita (2012)** — el núcleo discutido y verificado.
- **Goldstein (1999)** — el modelo predador-presa de fase fija que el paper refuta (blanco de
  los 90°), citado en la intro.
- **Raissi et al. (2019)** — el paper origen del método PINN que se usa, citado en métodos.

**Retiradas de la bibliografía** (no se discutían y/o no se citaban en el cuerpo): Goodwin
(1967) (LV es de manual; se nombra sin atribución), Kalecki (1935) y Kydland–Prescott (1982)
(caían con el párrafo de microfundación), y Almon (1965), Rackauckas et al. (2020) y
Ramsay–Hooker (2017) (entradas huérfanas, nunca citadas en el texto).
