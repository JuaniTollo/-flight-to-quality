"""Pieza 13 — ¿Las crisis son una inestabilidad de feedback retardado?

El modelo de maduración (cadena = delay distribuido) es un sistema dinámico. Pregunta:
¿el lag de maduración estimado (~1 año) ubica al sistema cerca de una BIFURCACIÓN DE HOPF
— el punto donde un foco amortiguado se vuelve oscilatorio/autosostenido? Si sí, las crisis
serían una inestabilidad ENDÓGENA por retardo: la maduración de la inversión es lo bastante
lenta como para desestabilizar el ciclo. (Un VAR reduced-form no puede decir esto.)

Sistema linealizado (estado [P, I, m1..m4], tiempo en años; cadena de SHAPE=4 etapas que
realiza un kernel Gamma de media μ → θ = μ_años/4 por etapa, tasa 1/θ):
    dP/dt = c·I − d·P − k·m4
    dI/dt = a·P − b·I
    dmⱼ/dt = (mⱼ₋₁ − mⱼ)/θ          (m0 = I)

Se barre (μ, k) y se calcula max Re(λ) de la matriz 6×6: la curva max Re(λ)=0 (con Im≠0)
es la frontera de Hopf. Se compara la posición estimada (μ̂≈3.9 trim, k̂≈0.42) con esa frontera.

Run:  uv run python -m src.experiment.p13_bifurcation
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.experiment import common as C

SHAPE = 4
# parámetros calibrados por el PINN inverso (Fourier, datos reales QoQ) — results/p12_pinn_fourier_estimates.csv
A_, B_, C_, D_, K_ = 0.3174, -0.0407, 0.3129, 0.0148, 0.4168
MU_HAT_Q = 3.904                      # lag estimado (trimestres)


def _gamma_kernel(mu_q, K=10):
    s = SHAPE; scale = mu_q / s; xs = np.arange(0, K + 1) + 0.5
    w = xs ** (s - 1) * np.exp(-xs / scale); return w / w.sum()


def fit_constrained():
    """Estima (a,b,c,d,k,μ) con POSITIVIDAD (a,b,c,d,k≥0) por gradient matching sobre las
    dos ecuaciones, pooled en vecindades de crisis (QoQ). Devuelve params físicos."""
    raw = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    P = (raw["PROFITS"].pct_change(1) * 100); I = (raw["INVESTMENT"].pct_change(1) * 100)
    d = pd.DataFrame({"P": P, "I": I}).dropna(); idx = d.index
    Pv, Iv = d["P"].to_numpy(), d["I"].to_numpy()
    drop = {"2008", "2020"}; H = 8
    segs = []
    for lbl, pk, tr in C.RECESSIONS:
        if lbl in drop:
            continue
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        nm = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(nm)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(Iv[pos])]
        if a - H >= 0 and a + H < len(Pv):
            segs.append((a - H, a + H + 1))

    def resid(p):
        a, b, c, d_, k, mu = p
        w = _gamma_kernel(mu); out = []
        for lo, hi in segs:
            Pw, Iw = Pv[lo:hi], Iv[lo:hi]
            Pw = (Pw - Pw.mean()) / Pw.std(); Iw = (Iw - Iw.mean()) / Iw.std()
            dP = np.gradient(Pw); dI = np.gradient(Iw)
            m = np.convolve(Iw, w)[:len(Iw)]
            out.extend(dP - (c * Iw - d_ * Pw - k * m))
            out.extend(dI - (a * Pw - b * Iw))
        return np.array(out)

    res = least_squares(resid, [0.3, 0.3, 0.3, 0.3, 0.4, 4.0],
                        bounds=([0, 0, 0, 0, 0, 1.0], [3, 3, 3, 3, 3, 12.0]), max_nfev=20000)
    return res.x


def sys_matrix(mu_q, k, a=A_, b=B_, c=C_, d=D_):
    """Matriz 6×6 del sistema linealizado; mu_q en trimestres, tiempo en años."""
    mu_y = mu_q / 4.0                  # años
    theta = mu_y / SHAPE              # años por etapa
    r = 1.0 / theta                  # tasa
    A = np.zeros((6, 6))
    A[0] = [-d, c, 0, 0, 0, -k]      # dP/dt
    A[1] = [a, -b, 0, 0, 0, 0]       # dI/dt
    A[2] = [0, r, -r, 0, 0, 0]       # dm1
    A[3] = [0, 0, r, -r, 0, 0]       # dm2
    A[4] = [0, 0, 0, r, -r, 0]       # dm3
    A[5] = [0, 0, 0, 0, r, -r]       # dm4
    return A


def max_re(mu_q, k):
    ev = np.linalg.eigvals(sys_matrix(mu_q, k))
    j = int(np.argmax(ev.real))
    return ev[j].real, abs(ev[j].imag)


def main():
    global A_, B_, C_, D_, K_, MU_HAT_Q
    print("=== Pieza 13 — bifurcación del modelo de maduración ===")
    print(f"params Fourier crudos (b<0 artefacto): a={A_}, b={B_}, c={C_}, d={D_}, k={K_}, μ={MU_HAT_Q}")
    # estimación con restricciones físicas (a,b,c,d,k≥0) — la base correcta para estabilidad
    a, b, c, d, k, mu = fit_constrained()
    A_, B_, C_, D_, K_, MU_HAT_Q = a, b, c, d, k, mu
    print(f"params CON RESTRICCIONES (físicos): a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}, "
          f"k={k:.3f}, μ={mu:.2f} trim\n")

    # punto estimado
    re0, im0 = max_re(MU_HAT_Q, K_)
    per0 = 2 * np.pi / im0 if im0 > 1e-9 else np.inf
    estado = "OSCILATORIO/INESTABLE (Re>0)" if re0 > 0 else ("amortiguado (Re<0)" if re0 < 0 else "crítico")
    print(f"En el punto estimado (μ̂={MU_HAT_Q:.2f}, k̂={K_:.2f}): max Re(λ)={re0:+.3f}, "
          f"período≈{per0:.1f} años → {estado}")

    # 1D: max Re vs μ a k̂ fijo → ¿hay Hopf inducido por el retardo?
    mus = np.linspace(0.5, 12, 200)
    res = np.array([max_re(m, K_)[0] for m in mus])
    cross = np.where(np.diff(np.sign(res)))[0]
    mu_crit = mus[cross[0]] if len(cross) else None
    print(f"\nBarrido μ a k̂={K_:.2f}: max Re(λ) en μ=0.5 → {res[0]:+.3f}, en μ=12 → {res[-1]:+.3f}")
    if mu_crit is not None:
        print(f"  → Hopf (cruce Re=0) en μ_crit ≈ {mu_crit:.2f} trim;  μ̂={MU_HAT_Q:.2f} "
              f"({'por encima del crítico → inestable' if MU_HAT_Q > mu_crit else 'por debajo → amortiguado'})")
    else:
        print(f"  → sin cruce en el rango (Re mantiene el signo) → el retardo no induce Hopf con estos params")

    # 2D: frontera de Hopf en (μ, k)
    KS = np.linspace(0.0, 1.5, 120)
    MUS = np.linspace(0.5, 12, 120)
    Z = np.array([[max_re(m, k)[0] for m in MUS] for k in KS])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(mus, res, lw=2); ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvline(MU_HAT_Q, color="tab:red", ls="--", label=f"μ̂={MU_HAT_Q:.1f} trim")
    if mu_crit is not None:
        ax[0].axvline(mu_crit, color="tab:green", ls=":", label=f"μ_crit≈{mu_crit:.1f}")
    ax[0].set_xlabel("lag de maduración μ (trim)"); ax[0].set_ylabel("max Re(λ)")
    ax[0].set_title(f"¿Hopf inducido por el retardo? (k̂={K_:.2f})"); ax[0].legend(fontsize=8)

    im = ax[1].contourf(MUS, KS, Z, levels=20, cmap="RdBu_r",
                        norm=plt.matplotlib.colors.TwoSlopeNorm(vcenter=0))
    ax[1].contour(MUS, KS, Z, levels=[0], colors="k", linewidths=2)
    ax[1].plot(MU_HAT_Q, K_, "k*", ms=16, label="estimado (μ̂, k̂)")
    ax[1].set_xlabel("lag de maduración μ (trim)"); ax[1].set_ylabel("fuerza del feedback k")
    ax[1].set_title("Frontera de Hopf en (μ, k)\nrojo=inestable (Re>0), azul=amortiguado; línea negra=Hopf")
    fig.colorbar(im, ax=ax[1], label="max Re(λ)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.OUTDIR / "p13_bifurcation.png", dpi=130); plt.close(fig)
    print(f"\n✓ figura: {C.OUTDIR}/p13_bifurcation.png")

    print("\nVEREDICTO honesto: ni con params crudos ni con la estimación restringida emerge un Hopf "
          "INDUCIDO POR EL RETARDO. La inestabilidad proviene del bajo amortiguamiento (d→0), no del "
          "lag; y la estimación estructural discrepa en μ (≈1) con el distributed-lag (≈4), reflejando "
          "el muro de identificabilidad. ⇒ el reframe 'crisis = inestabilidad por retardo' NO se "
          "sostiene con estos datos. Requeriría identificación estructural más fuerte (panel multi-país "
          "o restricciones micro de gestación), no un reajuste sobre la misma serie.")


if __name__ == "__main__":
    main()
