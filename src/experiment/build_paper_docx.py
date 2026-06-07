"""Genera una copia en .docx del paper de 5 páginas (paper/main.tex).

Replica el contenido (texto + figuras de paper/figures/) sin depender de pandoc/LaTeX.
Run:  uv run python -m src.experiment.build_paper_docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "main.docx"


def H(doc, t, lvl):
    doc.add_heading(t, level=lvl)


def P(doc, t, size=11, italic=False, bold=False, align=None):
    p = doc.add_paragraph()
    r = p.add_run(t); r.italic = italic; r.bold = bold; r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def fig(doc, name, cap, width=5.6):
    f = FIGS / name
    if f.exists():
        doc.add_picture(str(f), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = doc.add_paragraph(); r = c.add_run(cap); r.italic = True; r.font.size = Pt(9)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def main():
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    t = doc.add_heading(
        "A distributed maturation delay in the profit–investment cycle: "
        "differentiable inverse calibration and identifiability limits", level=0)
    P(doc, "Juan I. Tollo", size=11, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    H(doc, "Abstract", 1)
    P(doc, "We study the profit–investment cycle of the U.S. economy (quarterly, 1948–2026) "
           "through the lens of overaccumulation: past investment depresses current profitability "
           "with a delay. Using the cross-correlation function and a recession event study, we show "
           "a dominant contemporaneous co-movement together with a negative feedback centred about "
           "one year, which builds up before recessions—consistent with Tapia Granados (2012). "
           "Fixed-phase predator–prey oscillators (Lotka–Volterra, FitzHugh–Nagumo) cannot reproduce "
           "this shape. We instead pose a physical maturation-chain model, a distributed-delay ODE, "
           "and calibrate it by a differentiable inverse (physics-informed neural network with "
           "Fourier features). The inverse recovers the maturation lag cleanly on synthetic data "
           "(8.9% error) and estimates ≈1 year on real data; the lag is stable to transform choice "
           "and leave-one-crisis-out. Our central result is one of identifiability: the centre of the "
           "delay is robustly identified, but its distributional width is not, and the feedback is "
           "observationally equivalent to a linear VAR(p≥4) at the level of the second-order "
           "autocovariance.", italic=True)

    H(doc, "1. Introduction", 1)
    P(doc, "The Marxian tradition of overaccumulation (Astarita; Tapia Granados, 2023) holds that "
           "capital accumulation eventually depresses the rate of profit, generating endogenous "
           "cycles and crises. A long line of nonlinear models—starting from Goodwin's (1967) growth "
           "cycle, a Lotka–Volterra (LV) predator–prey system, and its empirical predator–prey "
           "estimate by Goldstein (1999)—formalises this as an antagonism between profits and a "
           "second variable. These models impose an instantaneous coupling with a fixed 90° phase lag.")
    P(doc, "We ask a narrower, mechanistic question: is the feedback from investment to profits an "
           "instantaneous antagonism, or a delayed one whose timing reflects the maturation "
           "(gestation) of investment? We find the latter. The empirical signature is a "
           "contemporaneous co-movement plus a negative feedback concentrated about one year out—a "
           "shape no fixed-phase oscillator can generate, but a distributed-delay system can. We "
           "formalise this as a physical maturation-chain ODE and calibrate it by a differentiable "
           "inverse problem. The contribution is descriptive and methodological: a transform-robust, "
           "leave-one-out-stable estimate of a structural maturation lag, recovered by an inverse "
           "method where classical single-shooting fails, together with an explicit characterisation "
           "of what cannot be identified from these data.")

    H(doc, "2. Data and methods", 1)
    P(doc, "We use quarterly U.S. corporate profits and gross private domestic investment (FRED, "
           "1948–2026, n=313). To avoid the spurious order-4 structure that the year-over-year "
           "transform injects at the lag of interest, we work in quarter-over-quarter growth "
           "(x_t = Δlog), which we verified roughly halves the apparent feedback magnitude vs YoY.")
    P(doc, "Descriptive. We compute the sample cross-correlation function (CCF) "
           "ρ(k) = corr(profits_t, investment_{t+k}) and, around the twelve NBER recessions, a "
           "superposed-epoch (event-study) composite aligned at the investment trough.")
    P(doc, "Physical model. Investment matures through n stages before depressing profits; by the "
           "linear-chain trick this realises a Gamma-distributed delay of mean μ:")
    P(doc, "    dP/dt = c·I − d·P − k·m,    dI/dt = a·P − b·I,    m(t) = Σ_j w_j(μ)·I(t−j),", size=10)
    P(doc, "with P profits, I investment, m matured investment, and w(μ) a Gamma kernel of mean μ "
           "(the maturation lag).")
    P(doc, "Differentiable inverse. We calibrate (a,b,c,d,k,μ) by an inverse physics-informed neural "
           "network (PINN): a network t↦[P,I] trained with a data loss on the observations and a "
           "physics-residual loss enforcing the ODE, optimising network weights and ODE parameters "
           "jointly (loss = L_data + λ·L_phys, λ=1; after Raissi et al., 2019; Rackauckas et al., "
           "2020). On long windows a plain network suffers spectral bias and the lag collapses; "
           "Fourier-feature inputs remove this. We validate recovery on synthetic data with a known "
           "lag, and assess identifiability by a profile over μ, a leave-one-crisis-out analysis, and "
           "the theoretical CCF of a fitted VAR(p) from the discrete Lyapunov equation.")

    H(doc, "3. Results", 1)
    P(doc, "The pattern: a delayed feedback, not an instantaneous antagonism. The CCF oscillates "
           "rather than decaying: a dominant contemporaneous peak (ρ(0)≈+0.6) and a negative lobe at "
           "lags 4–5 quarters (≈−0.3 in QoQ), returning to zero by lag 6 (Fig. 1). The event study "
           "(Fig. 2) shows profits turning down first and an investment–profit gap that builds up "
           "~1–1.5 years before the trough—the overaccumulation buildup documented by Tapia Granados "
           "(2012), here re-confirmed by an independent method. The effective phase is near "
           "co-movement (~10°), not the 90° of LV/FN; fixed-phase oscillators are accordingly refuted, "
           "as their CCF is a rigid cosine, not an isolated pulse plus a delayed negative lobe.")
    fig(doc, "p7_overaccumulation.png",
        "Figure 1. Cross-correlation profits↔investment: a strong contemporaneous co-movement and a "
        "delayed negative feedback (investment maturing ~1 year earlier depresses current profits). "
        "Robust to excluding 2008/2020.")
    fig(doc, "p8_composite_crisis.png",
        "Figure 2. Composite of NBER recessions aligned at the investment trough. Profits lead the "
        "downturn and the investment−profit gap builds up ~1–1.5 years before the trough.")
    P(doc, "The maturation lag is identifiable and stable. Five methods converge on a maturation lag "
           "of ≈4 quarters: the empirical kernel, the distributed-lag regression, a VAR(p≥4), a "
           "hierarchical pooled estimate (μ_pop≈4.7–4.9), and the differentiable inverse. The "
           "Fourier-feature inverse PINN recovers a known synthetic lag with 8.9% error (4.0→4.36 "
           "quarters) and estimates 3.90 quarters = 0.98 years on real data; the lag is stable across "
           "difference transforms (QoQ, Δlog) and to leave-one-crisis-out (span 0.62 quarter). The "
           "classical single-shooting and the latent-state inverse fail (lag collapses or biases by "
           "~55%); the well-posed convolution form with Fourier features succeeds.")
    P(doc, "What cannot be identified. The feedback explains only ~3% of the variance of profit "
           "growth (vs ~35% for the contemporaneous co-movement). Consequently the width of the delay "
           "distribution is not identifiable (flat likelihood profile; AIC favours a point delay), nor "
           "is its variation across crises. Moreover, the negative lobe is reproduced by the "
           "theoretical CCF of a linear VAR(p≥4) from its coefficients alone (Fig. 3): the "
           "distributed-delay kernel is observationally equivalent, at the level of the second-order "
           "autocovariance, to a linear stochastic system with longer memory. The maturation model is "
           "an interpretable reparameterisation of that linear content, not a mechanism separable from "
           "it with these data. A bifurcation analysis of the calibrated system does not yield a "
           "delay-induced Hopf instability, so we do not claim crises to be an endogenous "
           "delayed-feedback instability.")
    fig(doc, "p12_var_lyapunov_ccf.png",
        "Figure 3. Theoretical CCF of a fitted VAR(p) (Lyapunov) vs the empirical CCF. For p≥4 the "
        "linear model already reproduces the negative lobe at lags 4–5 from its coefficients, "
        "establishing observational equivalence with the distributed-delay kernel.", width=4.6)

    H(doc, "4. Discussion", 1)
    P(doc, "The substantive picture is consistent and honest. There is an endogenous overaccumulation "
           "feedback with a maturation lag of about one year, stronger near recessions, and we recover "
           "it with a physical, differentiable model where classical inversion fails. But two limits "
           "bound the claim. First, the economic content of the lag—its ~1-year centre and "
           "pre-recession buildup—is already in Tapia Granados (2012); our addition is the model and "
           "the inverse method, not the phenomenon. Second, with nine usable post-war crises and a "
           "~3% signal-to-variance ratio, the data cannot separate a genuinely distributed (stochastic) "
           "delay from a point delay, nor the maturation mechanism from a linear long-memory VAR. The "
           "microfoundation of a distributed delay is the aggregation of heterogeneous investment "
           "gestation lags (Kalecki, 1935; Kydland and Prescott, 1982), so a distributed kernel is the "
           "mechanical consequence of aggregation rather than a conjecture; but confirming its shape "
           "requires either a multi-country panel (many more episodes, to identify the width) or "
           "measured sectoral gestation times (to predict the macro kernel, a test a reduced-form VAR "
           "cannot pass). These are the natural next steps.")

    H(doc, "5. Conclusion", 1)
    P(doc, "The profit–investment overaccumulation feedback is well described by a distributed "
           "maturation delay centred at ~1 year, recovered by a differentiable inverse that overcomes "
           "the spectral bias and ill-posedness of classical calibration. The robust, transferable "
           "result is the location of the identifiability frontier: the delay's centre is recoverable "
           "and stable, while its width and its separation from a linear model are not—a boundary set "
           "by signal strength and episode count, not by tuning. Pushing past it requires new data, "
           "not a reparameterisation.")

    # ---------------- Apéndices ----------------
    def table(doc, header, rows):
        t = doc.add_table(rows=1, cols=len(header)); t.style = "Light Grid Accent 1"
        for j, h in enumerate(header):
            run = t.rows[0].cells[j].paragraphs[0].add_run(h); run.bold = True; run.font.size = Pt(9)
        for row in rows:
            cells = t.add_row().cells
            for j, v in enumerate(row):
                r = cells[j].paragraphs[0].add_run(v); r.font.size = Pt(9)

    H(doc, "Appendix A. Robustness and falsification battery", 1)
    P(doc, "Table A1 collects the auxiliary tests. The location of the lag (≈5 quarters) is stable "
           "within the difference family (QoQ, Δlog) and to leave-one-crisis-out (Fig. A1, left; "
           "pooled μ≈4.7, span 0.62), though per-crisis CIs are wide because identification emerges "
           "only by pooling. The negative lobe is not a clean crisis-specific signature: a kernel "
           "placebo finds it roughly as often off-crisis, and a VAR(p≥4) reproduces it from its "
           "coefficients (Fig. 3). YoY both inflates the magnitude ~2× and shifts the apparent lag "
           "(to ~2 quarters); two-sided cycle filters collapse it to a contemporaneous term — so we "
           "report QoQ throughout (Fig. A1, right).")
    table(doc, ["Test", "Question", "Result"], [
        ["Transform robustness", "lag robust to transform?", "μ*≈5q (QoQ/Δlog); YoY→2q; filters→1q"],
        ["Lag stability / LOO", "lag stable across crises?", "pooled μ≈4.7q; LOO span 0.62q; all in CI"],
        ["Hierarchical pooling", "population lag & spread", "μ_pop≈4.9q [4.4,5.5]; spread contested"],
        ["Kernel placebo", "is the 4–5q lobe crisis-specific?", "ubiquitous (appears off-crisis too)"],
        ["VAR(p) Lyapunov", "does a linear VAR reproduce it?", "yes for p≥4 ⇒ obs. equivalence"],
        ["Pooled-dynamics placebo", "is shared crisis dynamics special?", "preliminary: ratios ≈ crisis ⇒ not special"],
    ])
    P(doc, "Table A1. Auxiliary robustness and falsification tests.", size=9, italic=True)
    fig(doc, "p12_lag_stability.png",
        "Figure A1a. Per-crisis maturation lag with leave-one-crisis-out: the pooled lag does not "
        "depend on any single episode.", width=5.0)
    fig(doc, "p12_transform_robustness.png",
        "Figure A1b. Estimated lag location under different transforms: stable within the difference "
        "family; shifted by YoY and by two-sided cycle filters.", width=5.0)

    H(doc, "Appendix B. Inverse-PINN variants and a stability check", 1)
    P(doc, "The maturation lag is recoverable by the differentiable inverse only when the surrogate "
           "network can represent the series and the parameters are well-posed (Table B1). A plain "
           "network on the long window suffers spectral bias and the lag collapses; per-crisis "
           "pooling fixes the fit (R²=0.86) but not the lag; only Fourier-feature inputs recover the "
           "lag cleanly (8.9% synthetic error, 0.98 years real). A bifurcation analysis of the "
           "calibrated system (Fig. B1) finds no delay-induced Hopf instability: the instability comes "
           "from low damping, not the lag, and the structural fit disagrees with the distributed-lag "
           "estimate on μ. We therefore do not advance an 'endogenous delayed-feedback instability' "
           "claim — it would require the stronger identification a panel or micro gestation data "
           "could provide.")
    table(doc, ["Variant", "Synthetic (true 4q)", "Real lag", "Verdict"], [
        ["Single-shooting / convolution", "1.86q (54% err)", "collapses to 0.23q, non-physical", "fails"],
        ["Per-crisis pooled", "1.64q (59% err)", "0.84q (R²=0.86)", "fits, lag collapses"],
        ["Profile-likelihood in μ", "—", "inconclusive", "unstable"],
        ["Fourier features", "4.36q (8.9%)", "3.90q = 0.98yr (R²=1)", "recovers"],
    ])
    P(doc, "Table B1. Inverse-PINN variants; only Fourier-feature inputs both fit and recover the lag.",
      size=9, italic=True)
    fig(doc, "p13_bifurcation.png",
        "Figure B1. Stability of the calibrated maturation system: no delay-induced Hopf (left), and "
        "the Hopf boundary in (μ, k) with the estimate marked (right). The mechanistic 'instability' "
        "reading is not supported by these data.", width=5.6)

    H(doc, "References", 1)
    refs = [
        "Almon, S. (1965). The distributed lag between capital appropriations and expenditures. Econometrica 33(1), 178–196.",
        "Goldstein, J.P. (1999). Predator–prey model estimates of the cyclical profit squeeze. Metroeconomica 50(2), 139–173.",
        "Goodwin, R.M. (1967). A growth cycle. In Socialism, Capitalism and Economic Growth. Cambridge University Press.",
        "Kalecki, M. (1935). A macrodynamic theory of business cycles. Econometrica 3(3), 327–344.",
        "Kydland, F.E., Prescott, E.C. (1982). Time to build and aggregate fluctuations. Econometrica 50(6), 1345–1370.",
        "Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. Journal of Computational Physics 378, 686–707.",
        "Rackauckas, C., et al. (2020). Universal differential equations for scientific machine learning. arXiv:2001.04385.",
        "Ramsay, J., Hooker, G. (2017). Dynamic Data Analysis. Springer.",
        "Tapia Granados, J.A. (2012). Statistical evidence of falling profits as cause of recession. Review of Radical Political Economics 44(4), 484–493.",
    ]
    for r in refs:
        p = doc.add_paragraph(r); p.runs[0].font.size = Pt(9)

    doc.save(str(OUT))
    print(f"✓ {OUT}")


if __name__ == "__main__":
    main()
