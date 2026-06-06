"""Infraestructura compartida del experimento (rediseño 2026-06-03).

Un solo lugar para: datos YoY, fechado NBER, los osciladores (FN, LV) integrados con
scipy, normalización z-score anti-leakage y el ajuste por single-shooting.

Diseño deliberadamente determinista y en UN solo lenguaje (Python) para que todos los
modelos —osciladores y baselines lineales— se evalúen sobre EXACTAMENTE los mismos
objetivos (la lección del scope mismatch: comparar siempre targets alineados).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "lotka_volterra.csv"
OUTDIR = ROOT / "output" / "experiment"
OUTDIR.mkdir(parents=True, exist_ok=True)

COLS = ["PROFITS_YOY", "INVEST_YOY"]

# Recesiones NBER (pico, valle), fin de trimestre. Ciclo k = (valle_{k-1}, valle_k].
RECESSIONS = [
    ("1949", "1948-12-31", "1949-12-31"), ("1954", "1953-06-30", "1954-06-30"),
    ("1958", "1957-09-30", "1958-06-30"), ("1961", "1960-06-30", "1961-03-31"),
    ("1970", "1969-12-31", "1970-12-31"), ("1975", "1973-12-31", "1975-03-31"),
    ("1980", "1980-03-31", "1980-09-30"), ("1982", "1981-09-30", "1982-12-31"),
    ("1991", "1990-09-30", "1991-03-31"), ("2001", "2001-03-31", "2001-12-31"),
    ("2008", "2007-12-31", "2009-06-30"), ("2020", "2020-03-31", "2020-06-30"),
]
PEAKS = [pd.Timestamp(p) for _, p, _ in RECESSIONS]
TROUGHS = [pd.Timestamp(t) for _, _, t in RECESSIONS]
LABELS = [lbl for lbl, _, _ in RECESSIONS]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["DATE"]).set_index("DATE")
    return df[COLS].dropna()


# --------------------------------------------------------------------------- modelos
def fn_rhs(u, p):                      # p=[a,b,logc,I]; c=exp(logc)>0
    a, b, logc, I = p
    c = np.exp(logc)
    return np.array([c * (u[0] - u[0] ** 3 / 3 - u[1] + I), (u[0] - a - b * u[1]) / c])


def lv_rhs(u, p):                      # p=[alpha,beta,delta,gamma] (>0); estado en cuadrante +
    al, be, de, ga = p
    return np.array([al * u[0] - be * u[0] * u[1], de * u[0] * u[1] - ga * u[1]])


def lin_rhs(u, p):                     # oscilador LINEAL 2D: du/dt = A·u + c ; p=[a11,a12,a21,a22,c1,c2]
    return np.array([p[0] * u[0] + p[1] * u[1] + p[4], p[2] * u[0] + p[3] * u[1] + p[5]])


@dataclass
class Model:
    name: str
    rhs: Callable
    p0: np.ndarray
    needs_offset: bool                 # LV vive en el cuadrante + (desplaza el estado)
    guard: Callable                    # rechaza θ inválidos antes de integrar


FN = Model("FN", fn_rhs, np.array([0.2, 0.3, np.log(3.0), 0.0]), False,
           lambda p: abs(p[2]) <= 6)
LV = Model("LV", lv_rhs, np.array([1.0, 0.3, 0.4, 1.5]), True,
           lambda p: np.all(np.asarray(p) > 0))
LIN = Model("LIN", lin_rhs, np.array([-0.1, 0.3, -0.3, -0.1, 0.0, 0.0]), False, lambda p: True)


def integrate(model: Model, theta, u0, t_eval):
    """Integra el oscilador; devuelve array 2×n (filas: profits, investment) o None."""
    theta = np.asarray(theta, float)
    if not (np.all(np.isfinite(theta)) and model.guard(theta)):
        return None
    try:
        sol = solve_ivp(lambda t, u: model.rhs(u, theta), (t_eval[0], t_eval[-1]), u0,
                        t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-8)
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] != len(t_eval) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y


def fit(model: Model, t, V, R, p0=None, restarts=(1.0, 1.2, 0.8)):
    """Single-shooting: ajusta θ minimizando el error de trayectoria desde u0=(V0,R0)."""
    p0 = model.p0 if p0 is None else np.asarray(p0, float)
    u0 = np.array([V[0], R[0]])

    def loss(theta):
        y = integrate(model, theta, u0, t)
        if y is None:
            return 1e8
        return float(np.sum((y[0] - V) ** 2 + (y[1] - R) ** 2))

    best, best_f = p0.copy(), np.inf
    for s in restarts:
        res = minimize(loss, p0 * s, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if res.fun < best_f:
            best_f, best = res.fun, res.x
    return best, best_f


# --------------------------------------------------------------------------- z-score
def zscorer(train_df: pd.DataFrame, needs_offset: bool):
    """Devuelve (to_z, to_raw, offset) ajustados SOLO con train (anti-leakage).
    Para LV agrega un offset para que el estado quede en el cuadrante +."""
    mu = train_df[COLS].mean().to_numpy()
    sd = train_df[COLS].std().to_numpy()
    offset = 0.0
    if needs_offset:
        zmin = ((train_df[COLS].to_numpy() - mu) / sd).min()
        offset = -zmin + 1.0
    to_z = lambda raw: (raw - mu) / sd + offset           # raw (2,) o (n,2) -> z
    to_raw = lambda z: (z - offset) * sd + mu
    return to_z, to_raw, offset, mu, sd
