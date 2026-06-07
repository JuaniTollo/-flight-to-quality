"""Pieza 9 — ¿Hay UNA dinámica presa-depredador COMÚN a TODAS las crisis (pooling)?

Test falsable de fondo. Ajustar UNA crisis sola es trivial (una oscilación, cualquier
oscilador la calza). Lo que importa: ¿un MISMO vector de parámetros dinámicos θ_shared
describe VARIAS crisis a la vez, o cada crisis necesita los suyos?

Diseño:
  1. Ventana de cada crisis = ±6 trimestres del fondo de inversión (mismo ancla que p8).
     Z-score POR VENTANA (cada crisis comparable en escala; la dinámica se compara, no el nivel).
  2. Dos ajustes y su error TOTAL sobre todas las crisis:
     (a) INDEPENDIENTE — θ_c propio por crisis. Piso de error (sobreajusta).
     (b) POOLED      — un ÚNICO θ_shared (coeficientes presa-depredador) común a todas;
                        solo se permite u0_c por crisis (cond. inicial). Se minimiza la
                        SUMA de errores de trayectoria sobre todas las ventanas a la vez.
  3. Métrica clave: ratio SSE_pooled / SSE_indep.  ~1 → dinámica compartida;  >>1 → idiosincrásica.
     Reporta error por crisis bajo el pooled (¿cuáles rompen el pooling?).
  4. LV y FN. Figura: observado vs pooled superpuestos por crisis.

Caveats honestos: n chico por ventana (13 pts), identificabilidad, sensibilidad a
inicialización (varios restarts), u0 libre le da grados de libertad al pooled.

Run:  uv run python -m src.experiment.p9_pooled
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from src.experiment import common as C

HW = 6                      # media-ventana: ±6 trimestres del fondo
WIN = 2 * HW + 1            # 13 puntos por ventana
CAP = 1e3                   # |estado| > CAP -> trayectoria descartada (evita LSODA lento)


def integ(model, theta, u0, t):
    """Integración guardada y RÁPIDA para el optimizador.

    El single-shooting de common.py usa LSODA con tolerancias finas; en regiones de θ
    casi-rígidas LSODA puede tardar decenas de segundos por evaluación (verificado).
    Acá: RK45 con `max_step` acotado y rechazo temprano de trayectorias que explotan,
    para que cada evaluación cueste ~constante. Devuelve 2×n o None.
    """
    theta = np.asarray(theta, float)
    if not (np.all(np.isfinite(theta)) and model.guard(theta)):
        return None

    def f(tt, u):
        return model.rhs(u, theta)

    def blow(tt, u):                      # evento: corta si |estado| supera CAP
        return CAP - np.max(np.abs(u))
    blow.terminal = True
    blow.direction = -1

    try:
        sol = solve_ivp(f, (t[0], t[-1]), u0, t_eval=t, method="RK45",
                        rtol=1e-5, atol=1e-7, max_step=1.0, events=blow)
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y


# --------------------------------------------------------------------------- ventanas
def crisis_windows(df):
    """Para cada recesión: ancla = inversión más deprimida en su vecindad; ventana ±HW.
    Devuelve lista de (label, t (0..WIN-1), V_z, R_z) con z-score POR ventana."""
    idx = df.index
    p, i = df["PROFITS_YOY"].to_numpy(), df["INVEST_YOY"].to_numpy()
    out = []
    for lbl, pk, tr in C.RECESSIONS:
        pk, tr = pd.Timestamp(pk), pd.Timestamp(tr)
        win = (idx >= pk - pd.offsets.QuarterEnd(2)) & (idx <= tr + pd.offsets.QuarterEnd(4))
        pos = np.where(win)[0]
        if len(pos) == 0:
            continue
        a = pos[np.argmin(i[pos])]
        if a - HW < 0 or a + HW >= len(p):
            continue
        sl = slice(a - HW, a + HW + 1)
        V, R = p[sl].copy(), i[sl].copy()
        # z-score por ventana (cada serie a media 0, sd 1) -> comparables en escala
        V = (V - V.mean()) / V.std()
        R = (R - R.mean()) / R.std()
        t = np.arange(WIN, dtype=float)
        out.append((lbl, t, V, R))
    return out


def to_quadrant(V, R, margin=2.0):
    """LV vive en el cuadrante +. Desplaza la ventana z-scoreada a positivo."""
    sh = -min(V.min(), R.min()) + margin
    return V + sh, R + sh, sh


# --------------------------------------------------------------------------- ajuste indep
def fit_independent(model, windows, restarts):
    """θ_c Y u0_c propios por crisis = libertad TOTAL. Es el PISO de error: el modelo
    pooled (θ compartido, u0 libre) está estrictamente anidado en éste, así que el
    independiente nunca puede ajustar peor → ratio pooled/indep ≥ 1 por construcción.
    Devuelve SSE total y dict por-crisis (sse, theta, u0, pred)."""
    nd = len(model.p0)
    total = 0.0
    per = {}
    for lbl, t, V, R in windows:
        if model.needs_offset:
            Vq, Rq, sh = to_quadrant(V, R)
        else:
            Vq, Rq, sh = V, R, 0.0
        u0_0 = np.array([Vq[0], Rq[0]])

        def loss(x):                              # x = [θ_c (nd), u0_c (2)]
            theta, u0 = x[:nd], x[nd:]
            y = integ(model, theta, u0, t)
            return 1e8 if y is None else float(np.sum((y[0] - Vq) ** 2 + (y[1] - Rq) ** 2))

        best, bf = None, np.inf
        for s in restarts:
            x0 = np.concatenate([model.p0 * s, u0_0])
            res = minimize(loss, x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
            if res.fun < bf:
                bf, best = res.fun, res.x
        theta, u0 = best[:nd], best[nd:]
        y = integ(model, theta, u0, t)
        per[lbl] = dict(sse=bf, theta=theta, u0=u0, sh=sh, pred=y, V=Vq, R=Rq, t=t)
        total += bf
    return total, per


# --------------------------------------------------------------------------- ajuste pooled
class _Budget(Exception):
    pass


def fit_pooled(model, windows, restarts, indep_per, budget_s=90.0):
    """UN θ_shared común; por cada crisis solo se libera u0_c.

    Optimización CONJUNTA de x = [θ_shared (nd) , {u0_c (2) por crisis}] con Nelder-Mead,
    WARM-STARTED en (θ̄ de los ajustes independientes, u0_c de los independientes). El warm
    start arranca cerca del óptimo, así Nelder-Mead solo tiene que ajustar el θ COMÚN
    (sacrificando algo de ajuste por crisis). Tope de tiempo por arranque (best-so-far) para
    que el problema 26-dim no se cuelgue. Arranques: θ̄ indep, mediana indep, p0.
    """
    nd = len(model.p0)
    labels = [w[0] for w in windows]
    data = []
    for lbl, t, V, R in windows:
        if model.needs_offset:
            Vq, Rq, _ = to_quadrant(V, R)
        else:
            Vq, Rq = V, R
        data.append((lbl, t, Vq, Rq, np.array([Vq[0], Rq[0]])))
    nC = len(data)

    # u0 inicial = el u0 ajustado por el modelo independiente (su primer punto)
    u0_init = np.array([indep_per[l]["u0"] for l in labels]).ravel()

    def split(x):
        return x[:nd], x[nd:].reshape(nC, 2)

    def total_loss(x):
        th, u0s = split(x)
        if not (np.all(np.isfinite(th)) and model.guard(th)):
            return 1e9
        s = 0.0
        for (lbl, t, Vq, Rq, _), u0 in zip(data, u0s):
            y = integ(model, th, u0, t)
            if y is None:
                return 1e9
            s += float(np.sum((y[0] - Vq) ** 2 + (y[1] - Rq) ** 2))
        return s

    import time
    theta_stack = np.array([indep_per[l]["theta"] for l in labels])
    th_cands = [theta_stack.mean(0), np.median(theta_stack, 0), model.p0.copy()]

    best_x, best_f = None, np.inf
    for th0 in th_cands:
        if not model.guard(th0):
            th0 = model.p0.copy()
        x0 = np.concatenate([th0, u0_init])
        state = {"fx": np.inf, "x": x0.copy(), "t0": time.time()}

        def obj(x):
            f = total_loss(x)
            if f < state["fx"]:
                state["fx"], state["x"] = f, x.copy()
            if time.time() - state["t0"] > budget_s:
                raise _Budget
            return f

        try:
            minimize(obj, x0, method="Nelder-Mead",
                     options={"maxiter": 100000, "maxfev": 200000,
                              "xatol": 1e-6, "fatol": 1e-8})
        except _Budget:
            pass
        if state["fx"] < best_f:
            best_f, best_x = state["fx"], state["x"]

    th, u0s = split(best_x)
    per = {}
    total = 0.0
    for (lbl, t, Vq, Rq, _), u0 in zip(data, u0s):
        y = integ(model, th, u0, t)
        sse = 1e9 if y is None else float(np.sum((y[0] - Vq) ** 2 + (y[1] - Rq) ** 2))
        per[lbl] = dict(sse=sse, u0=u0, pred=y, V=Vq, R=Rq, t=t)
        total += sse
    return total, per, th


# --------------------------------------------------------------------------- figura
def plot_pooled(model, windows, pooled_per, theta, ratio):
    n = len(windows)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (lbl, t, V, R) in zip(axes, windows):
        d = pooled_per[lbl]
        Vq, Rq, y = d["V"], d["R"], d["pred"]
        ax.plot(t, Vq, "o", color="tab:blue", ms=3, label="profits obs")
        ax.plot(t, Rq, "s", color="tab:green", ms=3, label="invest obs")
        if y is not None:
            ax.plot(t, y[0], "-", color="tab:blue", lw=1.4)
            ax.plot(t, y[1], "-", color="tab:green", lw=1.4)
        ax.axvline(HW, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{lbl}  (SSE={d['sse']:.1f})", fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=6, loc="best")
    fig.suptitle(f"{model.name} POOLED — un θ_shared común a todas las crisis "
                 f"(solo u0 por crisis)\nlíneas = modelo pooled · marcadores = observado "
                 f"(z-score por ventana) · ratio SSE pooled/indep = {ratio:.2f}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = C.OUTDIR / f"p9_pooled_{model.name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- polish
def _polish_independent(model, windows, per_i, per_p, theta, nd):
    """Re-ajusta cada ventana independiente arrancando también desde la solución pooled,
    para que el piso no quede atrapado en óptimos locales peores que el pooled (ratio_c≥1)."""
    for lbl, t, V, R in windows:
        d = per_i[lbl]
        Vq, Rq = d["V"], d["R"]
        x0 = np.concatenate([theta, per_p[lbl]["u0"]])

        def loss(x):
            y = integ(model, x[:nd], x[nd:], t)
            return 1e8 if y is None else float(np.sum((y[0] - Vq) ** 2 + (y[1] - Rq) ** 2))

        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if res.fun < d["sse"]:
            y = integ(model, res.x[:nd], res.x[nd:], t)
            per_i[lbl] = dict(sse=res.fun, theta=res.x[:nd], u0=res.x[nd:], sh=d["sh"],
                              pred=y, V=Vq, R=Rq, t=t)
    return per_i


# --------------------------------------------------------------------------- main
def run_model(model, windows, restarts):
    import time
    print(f"\n{'='*70}\n  MODELO {model.name}\n{'='*70}", flush=True)
    t0 = time.time()
    sse_i, per_i = fit_independent(model, windows, restarts)
    print(f"  [indep listo en {time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    sse_p, per_p, theta = fit_pooled(model, windows, restarts, per_i, budget_s=75.0)
    print(f"  [pooled listo en {time.time()-t0:.0f}s]", flush=True)

    # PULIDO del piso: el pooled (θ compartido) está anidado en el independiente, así que
    # el independiente NUNCA debería ajustar peor. Si en alguna ventana el indep quedó en un
    # óptimo local peor que la solución pooled, lo re-ajustamos arrancando DESDE la solución
    # pooled (θ_shared, u0_pooled). Garantiza ratio_c ≥ 1 y un piso honesto (no inflado).
    nd = len(model.p0)
    per_i = _polish_independent(model, windows, per_i, per_p, theta, nd)
    sse_i = sum(per_i[w[0]]["sse"] for w in windows)
    ratio = sse_p / sse_i if sse_i > 0 else np.inf

    print(f"\n  SSE INDEPENDIENTE (piso, θ por crisis) = {sse_i:8.2f}")
    print(f"  SSE POOLED        (θ_shared único)     = {sse_p:8.2f}")
    print(f"  >>> RATIO pooled/indep = {ratio:5.2f}  "
          f"(~1 = dinámica compartida ; >>1 = idiosincrásica)")
    print(f"\n  θ_shared = {np.array2string(theta, precision=3)}")

    print(f"\n  {'crisis':>7} | {'SSE indep':>9} | {'SSE pooled':>10} | {'ratio_c':>7} | "
          f"{'NRMSE_pooled':>12}")
    print("  " + "-" * 60)
    rows = []
    for lbl, t, V, R in windows:
        si, sp = per_i[lbl]["sse"], per_p[lbl]["sse"]
        rc = sp / si if si > 0 else np.inf
        # NRMSE: RMSE de la trayectoria pooled normalizado por sd (los datos son z, sd~1 cada serie)
        nrmse = np.sqrt(sp / (2 * WIN))
        rows.append((lbl, si, sp, rc, nrmse))
        print(f"  {lbl:>7} | {si:9.2f} | {sp:10.2f} | {rc:7.2f} | {nrmse:12.3f}")

    # cuáles rompen el pooling: peor ratio_c y peor NRMSE
    worst = sorted(rows, key=lambda r: -r[3])[:3]
    print(f"\n  crisis que MÁS rompen el pooling (mayor ratio_c): "
          + ", ".join(f"{r[0]} ({r[3]:.1f}×)" for r in worst))

    path = plot_pooled(model, windows, per_p, theta, ratio)
    print(f"  figura -> {path}")
    return dict(ratio=ratio, sse_i=sse_i, sse_p=sse_p, theta=theta, rows=rows, fig=path)


def main():
    df = C.load()
    windows = crisis_windows(df)
    print(f"Ventanas de crisis utilizables (±{HW} trim del fondo, {WIN} pts c/u): "
          f"{len(windows)} -> {[w[0] for w in windows]}")
    restarts = (1.0, 1.2, 0.8)

    results = {}
    for model in (C.LV, C.FN):
        results[model.name] = run_model(model, windows, restarts)

    print(f"\n{'='*70}\n  VEREDICTO\n{'='*70}")
    for name, r in results.items():
        verdict = ("DINÁMICA COMPARTIDA plausible" if r["ratio"] < 2
                   else "PARCIAL" if r["ratio"] < 5
                   else "IDIOSINCRÁSICA (sin estructura compartida)")
        print(f"  {name}: ratio={r['ratio']:.2f}  -> {verdict}")


if __name__ == "__main__":
    main()
