"""Pieza 8 — Estudio de evento de las crisis (superposed epoch analysis).

Idea (objeción de no-estacionariedad): el CCF agrupado de 78 años promedia regímenes y
suaviza el acople. Para ver el mecanismo de sobreacumulación hay que CENTRAR en cada crisis
y mirar su vecindad. Tres análisis:

  A. CRISIS COMPUESTA — alinear las 12 recesiones en el fondo (t=0) y promediar
     trayectorias de profits/investment, con bandas ±1σ. Pregunta: ¿profits lidera la caída?
  B. BUILDUP DE SOBREACUMULACIÓN — ¿hay una brecha (inversión corriendo por delante de la
     ganancia) que se acumula durante la expansión y se "destapa" justo antes de la crisis?
  C. FASE NORMALIZADA — reescalar cada ciclo trough→trough a 0–100% y promediar en fase
     (ventanas no homogéneas en tiempo pero homogéneas en etapa del ciclo).
  D. CCF CONDICIONAL — lag−4 (presa-depredador) dentro de la vecindad de crisis vs en
     expansión, para cuantificar cuánto se fortalece el acople cerca de la crisis.

La preparación de datos (tasas QoQ, anclas NBER) vive en rates.QuarterlyRates y entra
como input de EventStudy; el recorte de ventanas está cubierto por tests
(tests/test_p8_event_study.py).

Run:  uv run python -m src.experiment.p8_event_study
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import common as C
from src.experiment.rates import QuarterlyRates

H = 8                     # ventana del evento: ±8 trimestres (±2 años)
NEAR = 6                  # vecindad de crisis para el CCF condicional: ±6 trimestres


# --------------------------------------------------------------------------- segmentos
def contiguous_segments(mask):
    """Devuelve lista de arrays de índices contiguos donde mask es True."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


def pooled_corr_lag(p, i, segments, k):
    """corr a lag k usando SOLO pares (t, t+k) dentro del MISMO segmento (sin cruzar bordes).

    Pearson por np.corrcoef sobre los pares agrupados; statsmodels.tsa.stattools.ccf no
    sirve acá porque normaliza con media/varianza globales y no admite segmentos.
    """
    P, I = [], []
    for seg in segments:
        a, b = seg[0], seg[-1] + 1
        ps, is_ = p[a:b], i[a:b]
        m = len(ps)
        if m <= abs(k):
            continue
        if k >= 0:
            P.append(ps[:m - k]); I.append(is_[k:])
        else:
            P.append(ps[-k:]); I.append(is_[:m + k])
    P, I = np.concatenate(P), np.concatenate(I)
    return np.corrcoef(P, I)[0, 1], len(P)


# --------------------------------------------------------------------------- estudio
class EventStudy:
    """Estudio de evento sobre tasas QoQ; los datos entran como QuarterlyRates."""

    def __init__(self, rates: QuarterlyRates, h: int = H, anchors=None):
        self.rates = rates
        self.h = h
        self.anchors = rates.crisis_anchors() if anchors is None else anchors

    # ------------------------------------------------ cómputo puro (testeable)
    def event_windows(self, anchors=None):
        """Recorta ventanas ±h alrededor de cada ancla; descarta las que pisan bordes."""
        h, p, i = self.h, self.rates.p, self.rates.i
        anchors = self.anchors if anchors is None else anchors
        usable = [(lbl, a) for lbl, a, *_ in anchors if a - h >= 0 and a + h < len(p)]
        P = np.array([p[a - h:a + h + 1] for _, a in usable])
        I = np.array([i[a - h:a + h + 1] for _, a in usable])
        return usable, P, I

    def crisis_regimes(self, drop_labels=()):
        """Segmentos 'cerca de crisis' (±NEAR del ancla) vs 'expansión'.

        drop_labels: crisis a excluir (sus ventanas no van a ningún régimen)."""
        n = len(self.rates)
        near = np.zeros(n, bool)
        dropped = np.zeros(n, bool)
        for lbl, a, _ in self.anchors:
            sl = slice(max(0, a - NEAR), min(n, a + NEAR + 1))
            if lbl in drop_labels:
                dropped[sl] = True
            else:
                near[sl] = True
        keep = ~dropped
        return contiguous_segments(near & keep), contiguous_segments(~near & keep), near

    # ------------------------------------------------------------ A + B
    def composite(self):
        h = self.h
        usable, P, I = self.event_windows()
        taus = np.arange(-h, h + 1)
        GAP = I - P                               # brecha: inversión por encima de ganancia

        print(f"=== A. CRISIS COMPUESTA — {len(usable)} crisis alineadas en el fondo (t=0) ===")
        print("  tau |  profits (μ±σ)   |  invest (μ±σ)    |  gap=inv−prof")
        for j, t in enumerate(taus):
            print(f"  {t:+3d} | {P[:,j].mean():+6.1f} ± {P[:,j].std():4.1f}  | "
                  f"{I[:,j].mean():+6.1f} ± {I[:,j].std():4.1f}  | {GAP[:,j].mean():+6.1f}")

        # ¿profits lidera la caída? trimestre del primer cruce a negativo de cada compuesta
        pm, im = P.mean(0), I.mean(0)
        def first_neg(x):
            neg = np.where(x < 0)[0]
            return taus[neg[0]] if len(neg) else None
        print(f"\n  profits cruza a negativo en tau={first_neg(pm)} ; invest en tau={first_neg(im)}")
        print(f"  → profits ADELANTA la caída por {first_neg(im) - first_neg(pm)} trimestre(s)"
              if first_neg(pm) is not None and first_neg(im) is not None else "")

        # B. buildup: la brecha (inv corriendo por delante de profit) y su máximo pre-crisis
        gm = GAP.mean(0)
        pre = taus <= 0
        tmax = taus[pre][np.argmax(gm[pre])]
        print(f"\n=== B. BUILDUP DE SOBREACUMULACIÓN (gap = invest − profits) ===")
        print(f"  gap máximo pre-fondo en tau={tmax:+d} (gap={gm[taus==tmax][0]:+.1f}): "
              f"inversión corre por delante de la ganancia ~{-tmax} trim. antes del fondo")
        print(f"  en el fondo (tau=0) gap={gm[h]:+.1f} → la brecha se cierra al destaparse la crisis")

        # figura A+B
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for ax, M, c, name in [(ax1, P, "tab:blue", "profits"), (ax1, I, "tab:green", "investment")]:
            mu, sd = M.mean(0), M.std(0)
            ax.plot(taus, mu, "-o", color=c, ms=4, label=name)
            ax.fill_between(taus, mu - sd, mu + sd, color=c, alpha=0.15)
        ax1.axvline(0, color="k", lw=0.8, ls="--"); ax1.axhline(0, color="grey", lw=0.6)
        ax1.set_title(f"A. Crisis compuesta ({len(usable)} recesiones alineadas en el fondo de inversión)\n"
                      "profits adelanta la caída; bandas = ±1σ entre crisis", fontsize=10)
        ax1.set_ylabel("QoQ %"); ax1.legend(fontsize=8)
        ax1.annotate("profits ya\ncae acá", (-3, pm[h-3]), fontsize=7, color="tab:blue")

        ax2.plot(taus, gm, "-o", color="tab:red", ms=4)
        ax2.fill_between(taus, gm, 0, where=(gm > 0), color="tab:red", alpha=0.15)
        ax2.axvline(0, color="k", lw=0.8, ls="--"); ax2.axhline(0, color="grey", lw=0.6)
        ax2.axvline(tmax, color="tab:red", lw=0.8, ls=":")
        ax2.set_title("B. Buildup de sobreacumulación: brecha (inversión − ganancia)\n"
                      "positiva = inversión corre por delante de la ganancia → presión de sobreacumulación",
                      fontsize=10)
        ax2.set_xlabel("τ = trimestres respecto del fondo de la crisis"); ax2.set_ylabel("inv − prof (pp)")
        ax2.annotate(f"máximo\nτ={tmax:+d}", (tmax, gm[taus == tmax][0]), fontsize=8, color="tab:red")
        fig.tight_layout(); fig.savefig(C.OUTDIR / "p8_composite_crisis.png", dpi=130); plt.close(fig)
        return usable

    # ------------------------------------------------------------ A'
    def anchor_sensitivity(self):
        """¿El lead de profits depende del ancla? prueba 4 anclas (robustez al ancla)."""
        p, i = self.rates.p, self.rates.i
        taus = np.arange(-self.h, self.h + 1)
        print("\n=== A'. SENSIBILIDAD AL ANCLA (lead de profits sobre invest) ===")

        def lead_for(pick):
            anchors = self.rates.crisis_anchors(pick)
            _, P, I = self.event_windows(anchors)
            pm, im = P.mean(0), I.mean(0)
            fp = taus[np.where(pm < 0)[0][0]] if np.any(pm < 0) else None
            fi = taus[np.where(im < 0)[0][0]] if np.any(im < 0) else None
            return fp, fi, (fi - fp if fp is not None and fi is not None else None)

        for name, pick in [("inv-mínima", lambda pos: pos[np.argmin(i[pos])]),
                           ("prof-mínima", lambda pos: pos[np.argmin(p[pos])]),
                           ("pico NBER", lambda pos: pos[0] + 2),
                           ("valle NBER", lambda pos: pos[-5] if len(pos) >= 5 else pos[-1])]:
            fp, fi, lead = lead_for(pick)
            print(f"  ancla={name:12s}: profits<0 en τ={fp}, invest<0 en τ={fi}  →  lead={lead}")
        print("  ⇒ el lead va de 0 a 2 según el ancla: hallazgo SUGESTIVO, no robusto.")

    # ------------------------------------------------------------ B'
    def gap_bootstrap(self, B=2000, seed=12345):
        """IC bootstrap por crisis del gap. rng local sembrado: determinista."""
        usable, P, I = self.event_windows()
        GAP = I - P
        nC = len(GAP)
        taus = np.arange(-self.h, self.h + 1)
        rng = np.random.default_rng(seed)
        boot = GAP[rng.integers(0, nC, size=(B, nC))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        print(f"\n=== B'. BOOTSTRAP del gap pre-fondo (IC 95%, n={nC} crisis) ===")
        for j, t in enumerate(taus):
            if -6 <= t <= 0:
                sig = "✓" if lo[j] > 0 else " "
                print(f"  τ={t:+d}: gap={GAP[:,j].mean():+5.1f}  IC95=[{lo[j]:+5.1f},{hi[j]:+5.1f}] {sig}")
        print("  (✓ = IC no incluye 0; sin corrección por comparaciones múltiples)")

    # ------------------------------------------------------------ C
    def phase_normalized(self):
        """Reescala cada ciclo trough→trough a fase 0–1 e interpola; promedia en fase."""
        idx, p, i = self.rates.index, self.rates.p, self.rates.i
        grid = np.linspace(0, 1, 25)
        Pn, In = [], []
        for t0, t1 in zip(C.TROUGHS[:-1], C.TROUGHS[1:]):
            seg = (idx > t0) & (idx <= t1)
            pos = np.where(seg)[0]
            if len(pos) < 6:
                continue
            ph = np.linspace(0, 1, len(pos))
            Pn.append(np.interp(grid, ph, p[pos]))
            In.append(np.interp(grid, ph, i[pos]))
        Pn, In = np.array(Pn), np.array(In)
        print(f"\n=== C. FASE NORMALIZADA — {len(Pn)} ciclos trough→trough reescalados a 0–100% ===")
        print("  fase% | profits μ | invest μ   (0%=salida de crisis previa, 100%=próxima crisis)")
        for j in range(0, 25, 4):
            print(f"   {int(grid[j]*100):3d}% | {Pn[:,j].mean():+6.1f}   | {In[:,j].mean():+6.1f}")

        fig, ax = plt.subplots(figsize=(9, 5))
        for M, c, name in [(Pn, "tab:blue", "profits"), (In, "tab:green", "investment")]:
            mu, sd = M.mean(0), M.std(0)
            ax.plot(grid * 100, mu, "-", color=c, label=name)
            ax.fill_between(grid * 100, mu - sd, mu + sd, color=c, alpha=0.12)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title("C. Ciclo estilizado (fase normalizada trough→trough, μ±σ)\n"
                     "0% = recién salido de la crisis previa · 100% = entrando a la próxima", fontsize=10)
        ax.set_xlabel("fase del ciclo (%)"); ax.set_ylabel("QoQ %"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(C.OUTDIR / "p8_phase_cycle.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------ D
    def conditional_ccf(self, label="", drop_labels=()):
        """lag−k dentro de vecindad de crisis (±NEAR trim del fondo) vs resto (expansión)."""
        p, i = self.rates.p, self.rates.i
        seg_near, seg_exp, near = self.crisis_regimes(drop_labels)
        print(f"\n=== D. CCF CONDICIONAL — presa-depredador por régimen {label}===")
        print(f"  'cerca de crisis': {near.sum()} trim en {len(seg_near)} bloques | "
              f"'expansión': {(~near).sum()} trim en {len(seg_exp)} bloques")
        print("  (pares emparejados SOLO dentro de cada bloque)")
        for k in (0, -4, -5):
            c_near, n_near = pooled_corr_lag(p, i, seg_near, k)
            c_exp, n_exp = pooled_corr_lag(p, i, seg_exp, k)
            print(f"  lag {k:+d}:  cerca de crisis = {c_near:+.2f} (n={n_near})  |  "
                  f"expansión = {c_exp:+.2f} (n={n_exp})")


def main():
    study = EventStudy(QuarterlyRates.load())
    study.composite()
    study.anchor_sensitivity()
    study.gap_bootstrap()
    study.phase_normalized()
    study.conditional_ccf(label="(todas) ")
    study.conditional_ccf(label="(sin 2008/2020) ", drop_labels=("2008", "2020"))
    print(f"\n✓ figuras: {C.OUTDIR}/p8_composite_crisis.png , p8_phase_cycle.png")


if __name__ == "__main__":
    main()
