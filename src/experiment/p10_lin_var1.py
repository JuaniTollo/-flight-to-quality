"""Pieza 10 — Oscilador lineal 2D = VAR(1)  (LIN: du/dt = A u).

OBJETIVO (evaluacion critica de UNA clase de modelo): el oscilador LINEAL 2D es,
en tiempo discreto, exactamente un VAR(1):  u_{t} = M u_{t-1} + e_t,  con
M = exp(A·Δt).  Un foco amortiguado (autovalores complejos conjugados con |λ|<1)
produce oscilaciones decrecientes de periodo  T = 2π/θ  (θ = arg del autovalor).

PREGUNTA: un foco amortiguado de periodo ~9 trimestres, ¿reproduce la CCF empirica
(QoQ, cerca de crisis sin 2008/2020): lag0=+0.62, joroba NEGATIVA en lag4=-0.32,
lag5=-0.35)?  Comparamos la CCF TEORICA del VAR(1) ajustado contra la empirica.

CRITICO: la respuesta de impulso de un VAR(1) es  M^h  → un kernel GEOMETRICO /
oscilante-amortiguado, totalmente determinado por 4 numeros (la matriz 2x2). No tiene
grados de libertad para poner una JOROBA aislada en lag 4-5 sin co-movimiento fuerte
en los lags intermedios 1-3. Aca medimos cuanto le alcanza eso y cuanto le falta.

Transform PRINCIPAL: QoQ = pct_change(1) sobre NIVELES (PROFITS, INVESTMENT).
n = 9 crisis utilizables (sin 2008/2020). Ventana = ±6 trim del fondo de inversion.

Run:  uv run python -m src.experiment.p10_lin_var1
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import logm

from src.experiment import common as C

HALF = 6                       # ±6 trim → ventana de 13 trim (= P8/P9)
WIN = 2 * HALF + 1
KMAX = 8                       # lags de la CCF a inspeccionar
DROP = ("2008", "2020")        # excluir GFC y COVID (consigna)


# --------------------------------------------------------------------------- datos QoQ
def qoq(df):
    """QoQ = pct_change(1) sobre los NIVELES. Devuelve (df_qoq, p, i)."""
    lv = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    q = lv[["PROFITS", "INVESTMENT"]].pct_change(1).dropna() * 100.0
    q.columns = ["P", "I"]
    return q


def crisis_anchor_positions(idx, p_lvl, i_lvl):
    """Para cada recesion NBER: posicion del fondo de inversion en su vecindad (= P8/P9)."""
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        out.append((lbl, pos[np.argmin(i_lvl[pos])]))
    return out


# --------------------------------------------------------------------------- VAR(1)
def fit_var1(X):
    """OLS VAR(1):  X_t = c + M X_{t-1} + e_t.  X es (n,2) [P, I].
    Devuelve M (2x2), c (2,), Sigma (cov de residuos), n_eff."""
    Y = X[1:]                  # (n-1, 2)
    Z = X[:-1]                 # (n-1, 2)
    # design [1, P_{t-1}, I_{t-1}]
    D = np.column_stack([np.ones(len(Z)), Z])
    B, *_ = np.linalg.lstsq(D, Y, rcond=None)   # (3,2): fila0=c, filas1-2=M^T
    c = B[0]
    M = B[1:].T                # (2,2)
    resid = Y - D @ B
    Sigma = resid.T @ resid / max(1, len(Y) - 3)
    return M, c, Sigma, len(Y)


def eig_diag(M, dt=1.0):
    """Autovalores discretos de M → periodo y amortiguamiento.
    Tambien pasa a tiempo continuo A = logm(M)/dt para el lenguaje du/dt = A u."""
    ev = np.linalg.eigvals(M)
    # ordenar por modulo desc
    ev = ev[np.argsort(-np.abs(ev))]
    lam = ev[0]
    mod = abs(lam)
    if np.iscomplex(lam) and abs(lam.imag) > 1e-9:
        theta = abs(np.angle(lam))            # rad por paso
        period = 2 * np.pi / theta            # trimestres
        kind = "foco" + (" amortiguado" if mod < 1 else " explosivo" if mod > 1 else " neutro")
    else:
        period = np.inf
        kind = "nodo (real)"
    # half-life del modulo dominante: mod^h = 0.5
    half_life = np.log(0.5) / np.log(mod) if 0 < mod < 1 else np.inf
    # continuo (puede fallar si M tiene eig negativos reales)
    try:
        A = np.real(logm(M)) / dt
        ac = np.linalg.eigvals(A)
        ac = ac[np.argsort(-ac.real)]
    except Exception:
        A, ac = None, None
    return dict(eig=ev, mod=mod, period=period, half_life=half_life, kind=kind, A=A, ac=ac)


def theoretical_ccf(M, Sigma, kmax=KMAX):
    """CCF estacionaria implicada por el VAR(1):  X_t = M X_{t-1} + e_t, Cov(e)=Sigma.
    Γ(0) resuelve  Γ0 = M Γ0 M^T + Sigma  (Lyapunov discreto, via vec).
    Γ(k) = M^k Γ0  para k>=0 ;  corr_{P,I}(lag k) sigue la MISMA convencion que la
    CCF empirica del proyecto: lag k = corr(P_t, I_{t+k}) (I adelantado k respecto P).
    """
    # Lyapunov discreto: vec(Γ0) = (I - M⊗M)^{-1} vec(Sigma)
    n = M.shape[0]
    K = np.kron(M, M)
    G0 = np.linalg.solve(np.eye(n * n) - K, Sigma.reshape(-1)).reshape(n, n)
    G0 = 0.5 * (G0 + G0.T)     # simetrizar numericamente
    sP, sI = np.sqrt(G0[0, 0]), np.sqrt(G0[1, 1])
    # Cross-cov a lag k:
    #   k>=0:  E[X_{t+k} X_t^T] = M^k Γ0           → (Γk)[0,1] = Cov(P_{t+k}, I_t)
    #   queremos corr(P_t, I_{t+k}) = Cov(P_t, I_{t+k}).
    #   Cov(P_t, I_{t+k}) para k>0 = E[X_t X_{t+k}^T][0,1] = (Γ0 (M^k)^T)[0,1]
    #   para k<0 (I rezagado): = (M^{|k|} Γ0)[0,1]
    lags = np.arange(-kmax, kmax + 1)
    ccf = np.empty(len(lags))
    for j, k in enumerate(lags):
        if k >= 0:
            Mk = np.linalg.matrix_power(M, k)
            cov = (G0 @ Mk.T)[0, 1]            # Cov(P_t, I_{t+k})
        else:
            Mk = np.linalg.matrix_power(M, -k)
            cov = (Mk @ G0)[0, 1]              # Cov(P_t, I_{t+k}) con k<0
        ccf[j] = cov / (sP * sI)
    return lags, ccf, G0


# --------------------------------------------------------------------------- CCF empirica
def empirical_ccf_pooled(q, anchors, n_lvl, kmax=KMAX, drop=DROP):
    """CCF empirica pooled cerca de crisis (±6), pares SOLO dentro de cada ventana.
    Convencion: lag k = corr(P_t, I_{t+k})."""
    P_all, I_all = q["P"].to_numpy(), q["I"].to_numpy()
    # ojo: q tiene 1 fila menos que niveles (pct_change(1)). anchors estan en indice de
    # niveles; la fila t de niveles ~ fila t-1 de q. Reanclamos por fecha.
    segs = []
    for lbl, a in anchors:
        if lbl in drop:
            continue
        lo, hi = a - HALF, a + HALF + 1
        if lo < 1 or hi > n_lvl:             # qoq empieza en t=1 de niveles
            continue
        # mapear a indices de q: nivel t → q fila t-1
        segs.append((lo - 1, hi - 1))
    lags = np.arange(-kmax, kmax + 1)
    ccf = np.empty(len(lags)); ns = np.empty(len(lags), int)
    for j, k in enumerate(lags):
        Ps, Is = [], []
        for a0, b0 in segs:
            ps, is_ = P_all[a0:b0], I_all[a0:b0]
            m = len(ps)
            if m <= abs(k):
                continue
            if k >= 0:
                Ps.append(ps[:m - k]); Is.append(is_[k:])
            else:
                Ps.append(ps[-k:]);    Is.append(is_[:m + k])
        Ps, Is = np.concatenate(Ps), np.concatenate(Is)
        Ps = (Ps - Ps.mean()); Is = (Is - Is.mean())
        ccf[j] = float(np.corrcoef(Ps, Is)[0, 1]); ns[j] = len(Ps)
    return lags, ccf, ns, segs


# --------------------------------------------------------------------------- main
def main():
    q = qoq(None)
    lv = pd.read_csv(C.DATA, parse_dates=["DATE"]).set_index("DATE")
    p_lvl = lv["PROFITS"].to_numpy(); i_lvl = lv["INVESTMENT"].to_numpy()
    anchors = crisis_anchor_positions(lv.index, p_lvl, i_lvl)
    n_lvl = len(lv)

    print("=" * 78)
    print("Pieza 10 — LIN = VAR(1) (du/dt = A u). Foco amortiguado vs CCF empirica.")
    print(f"Transform: QoQ = pct_change(1) sobre NIVELES. Ventana ±{HALF} trim. Sin {DROP}.")
    print("=" * 78)

    # --- 1. VAR(1) en la SERIE COMPLETA (QoQ) ---
    X_full = q[["P", "I"]].to_numpy()
    Mf, cf, Sf, nf = fit_var1(X_full)
    df_full = eig_diag(Mf)
    print("\n### 1. VAR(1) en la SERIE COMPLETA (QoQ, n_obs={}) ###".format(nf))
    print("M (1 paso) =\n", np.array2string(Mf, precision=3, suppress_small=True))
    print(f"  autovalores: {np.array2string(df_full['eig'], precision=3)}")
    print(f"  |λ_dom| = {df_full['mod']:.3f}  ({df_full['kind']})")
    print(f"  periodo = {df_full['period']:.2f} trim   half-life modulo = {df_full['half_life']:.2f} trim")
    lags, ccf_full, G0f = theoretical_ccf(Mf, Sf)

    # --- 2. VAR(1) por VENTANA DE CRISIS (QoQ) + pooled stack ---
    print("\n### 2. VAR(1) por VENTANA DE CRISIS (±6 trim, QoQ) ###")
    print(f"{'crisis':6s} | {'|λ|':>5s} | {'periodo':>8s} | {'half-life':>9s} | tipo")
    per_win = {}
    stacks = []                                # para VAR pooled (apilar pares dentro de ventana)
    for lbl, a in anchors:
        if lbl in DROP:
            continue
        lo, hi = a - HALF, a + HALF + 1
        if lo < 1 or hi > n_lvl:
            continue
        Xw = q[["P", "I"]].to_numpy()[lo - 1:hi - 1]   # mapear niveles→q
        if len(Xw) < 5:
            continue
        Mw, cw, Sw, nw = fit_var1(Xw)
        dw = eig_diag(Mw)
        per_win[lbl] = dw
        per = f"{dw['period']:.1f}" if np.isfinite(dw['period']) else "∞(nodo)"
        hl = f"{dw['half_life']:.1f}" if np.isfinite(dw['half_life']) else "∞"
        print(f"{lbl:6s} | {dw['mod']:5.2f} | {per:>8s} | {hl:>9s} | {dw['kind']}")
        # apilar para pooled: usar pares (t-1 -> t) DENTRO de la ventana
        Xs = q[["P", "I"]].to_numpy()[lo - 1:hi - 1]
        stacks.append((Xs[:-1], Xs[1:]))

    periods = np.array([d["period"] for d in per_win.values() if np.isfinite(d["period"])])
    mods = np.array([d["mod"] for d in per_win.values()])
    n_focus = sum(1 for d in per_win.values() if "foco" in d["kind"])
    print(f"\n  resumen ventanas: {len(per_win)} crisis; focos (eig complejos) = {n_focus}/{len(per_win)}")
    if len(periods):
        print(f"  periodo (focos): mediana={np.median(periods):.1f}  media={np.mean(periods):.1f}  "
              f"rango=[{periods.min():.1f},{periods.max():.1f}] trim")
    print(f"  |λ_dom|: mediana={np.median(mods):.2f}  rango=[{mods.min():.2f},{mods.max():.2f}]")

    # --- 2b. VAR(1) POOLED (un M compartido, pares apilados dentro de ventanas) ---
    Zp = np.vstack([z for z, _ in stacks]); Yp = np.vstack([y for _, y in stacks])
    Dp = np.column_stack([np.ones(len(Zp)), Zp])
    Bp, *_ = np.linalg.lstsq(Dp, Yp, rcond=None)
    Mp = Bp[1:].T
    residp = Yp - Dp @ Bp
    Sp = residp.T @ residp / max(1, len(Yp) - 3)
    dp = eig_diag(Mp)
    print("\n### 2b. VAR(1) POOLED de crisis (un M compartido, {} pares) ###".format(len(Yp)))
    print("M_pooled =\n", np.array2string(Mp, precision=3, suppress_small=True))
    print(f"  |λ_dom|={dp['mod']:.3f} ({dp['kind']}), periodo={dp['period']:.2f} trim, "
          f"half-life={dp['half_life']:.2f} trim")
    _, ccf_pool, _ = theoretical_ccf(Mp, Sp)

    # --- 3. CCF EMPIRICA pooled cerca de crisis ---
    lags_e, ccf_emp, ns_e, segs = empirical_ccf_pooled(q, anchors, n_lvl)
    print("\n### 3. CCF EMPIRICA (QoQ, pooled cerca de crisis sin 2008/2020) ###")
    print(f"  ventanas usadas: {len(segs)}")

    # --- 4. COMPARACION CCF: empirica vs VAR(1) full vs VAR(1) pooled ---
    print("\n### 4. CCF empirica vs VAR(1) teorica  [lag k = corr(P_t, I_{t+k})] ###")
    print(f"{'lag':>4s} | {'empirica':>9s} | {'VAR1 full':>9s} | {'VAR1 pool':>9s}")
    foc = [0, 1, 2, 3, 4, 5]
    for j, k in enumerate(lags):
        mark = "  <--" if k in (0, 4, 5) else ""
        print(f"{k:+4d} | {ccf_emp[j]:+9.2f} | {ccf_full[j]:+9.2f} | {ccf_pool[j]:+9.2f}{mark}")

    # metricas de ajuste de la CCF en lags 0..KMAX (donde esta el patron)
    pos = lags >= 0
    def fitstats(theo):
        e = ccf_emp[pos]; t = theo[pos]
        rmse = float(np.sqrt(np.mean((e - t) ** 2)))
        corr = float(np.corrcoef(e, t)[0, 1])
        return rmse, corr
    rmse_f, corr_f = fitstats(ccf_full)
    rmse_p, corr_p = fitstats(ccf_pool)
    print(f"\n  ajuste CCF (lags 0..{KMAX}):  VAR1 full RMSE={rmse_f:.3f} corr={corr_f:+.2f} | "
          f"VAR1 pool RMSE={rmse_p:.3f} corr={corr_p:+.2f}")

    # diagnostico de la JOROBA: ¿el VAR pone minimo en 4-5? ¿hay co-mov en 1-3?
    def hump_diag(theo, name):
        argmin = lags[lags >= 0][np.argmin(theo[lags >= 0])]
        lag0 = theo[lags == 0][0]
        l4 = theo[lags == 4][0]; l5 = theo[lags == 5][0]
        l13 = theo[(lags >= 1) & (lags <= 3)]
        print(f"  [{name}] lag0={lag0:+.2f}  lag4={l4:+.2f}  lag5={l5:+.2f}  "
              f"min en lag={argmin}  |  lags1-3 medio={l13.mean():+.2f} (rango "
              f"[{l13.min():+.2f},{l13.max():+.2f}])")
    print("\n  -- diagnostico de la JOROBA negativa en 4-5 --")
    hump_diag(ccf_emp, "empirica ")
    hump_diag(ccf_full, "VAR1 full")
    hump_diag(ccf_pool, "VAR1 pool")

    # --- 5. respuesta de impulso (M^h) para mostrar el kernel geometrico ---
    print("\n### 5. KERNEL DE LAG = respuesta I→P (elemento [P,I] de M^h, h=0..KMAX) ###")
    print("  (cuanto se mueve P hoy por un shock de I hace h trimestres — kernel del VAR)")
    irf_full = [np.linalg.matrix_power(Mf, h)[0, 1] for h in range(KMAX + 1)]
    irf_pool = [np.linalg.matrix_power(Mp, h)[0, 1] for h in range(KMAX + 1)]
    print("  h:        " + " ".join(f"{h:>6d}" for h in range(KMAX + 1)))
    print("  full M^h: " + " ".join(f"{v:+6.2f}" for v in irf_full))
    print("  pool M^h: " + " ".join(f"{v:+6.2f}" for v in irf_pool))

    # --- CSV ---
    out = pd.DataFrame({"lag": lags, "ccf_emp": ccf_emp, "ccf_var1_full": ccf_full,
                        "ccf_var1_pool": ccf_pool, "n_emp": ns_e})
    out_csv = C.OUTDIR / "p10_lin_var1.csv"
    out.to_csv(out_csv, index=False)

    # --- figura ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.axhline(0, color="grey", lw=0.6)
    ax.plot(lags, ccf_emp, "-o", color="k", ms=5, lw=1.8, label="empirica (QoQ, crisis)")
    ax.plot(lags, ccf_full, "-s", color="tab:blue", ms=4, alpha=0.8, label=f"VAR(1) full (T={df_full['period']:.1f})")
    ax.plot(lags, ccf_pool, "-^", color="tab:red", ms=4, alpha=0.8, label=f"VAR(1) pooled crisis (T={dp['period']:.1f})")
    ax.axvspan(3.5, 5.5, color="orange", alpha=0.12)
    ax.set_xlabel("lag k  [corr(P_t, I_{t+k})]"); ax.set_ylabel("correlacion")
    ax.set_title("CCF empirica vs VAR(1) teorica\njoroba neg. empirica en lag 4-5 (banda)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.axhline(0, color="grey", lw=0.6)
    hh = np.arange(KMAX + 1)
    ax.plot(hh, irf_full, "-s", color="tab:blue", label="VAR(1) full")
    ax.plot(hh, irf_pool, "-^", color="tab:red", label="VAR(1) pooled crisis")
    ax.set_xlabel("h (trimestres)"); ax.set_ylabel("respuesta P a shock I  (M^h)[P,I]")
    ax.set_title("Kernel de lag del VAR(1): geometrico/amortiguado\n(no una joroba aislada en 4-5)", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Pieza 10 — LIN=VAR(1): foco amortiguado vs patron de lag distribuido empirico", fontsize=12)
    fig.tight_layout()
    out_fig = C.OUTDIR / "p10_lin_var1.png"
    fig.savefig(out_fig, dpi=130); plt.close(fig)

    print(f"\n✓ CSV:    {out_csv}")
    print(f"✓ figura: {out_fig}")


if __name__ == "__main__":
    main()
