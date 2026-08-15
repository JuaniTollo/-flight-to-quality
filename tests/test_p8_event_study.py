"""Tests del recorte de datos del estudio de evento (P8).

Cubren: la transformación QoQ de QuarterlyRates, la ventana de búsqueda de anclas
NBER (bordes incluidos), el descarte de ventanas que pisan los bordes de la muestra,
la partición en regímenes crisis/expansión y el emparejamiento de pares SOLO dentro
de cada segmento en pooled_corr_lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiment.p8_event_study import (EventStudy, contiguous_segments,
                                           pooled_corr_lag)
from src.experiment.rates import COL_INVEST, COL_PROFITS, QuarterlyRates


def make_rates(p, i, start="1947-03-31"):
    idx = pd.date_range(start, periods=len(p), freq="QE")
    return QuarterlyRates(pd.DataFrame({COL_PROFITS: p, COL_INVEST: i}, index=idx))


# --------------------------------------------------------------- QuarterlyRates.load
def test_load_computes_qoq_log_rates(tmp_path):
    idx = pd.date_range("1947-03-31", periods=5, freq="QE")
    prof = np.array([100.0, 110.0, 105.0, 120.0, 118.0])
    inv = np.array([80.0, 82.0, 90.0, 85.0, 95.0])
    csv = tmp_path / "lv.csv"
    pd.DataFrame({"DATE": idx, "PROFITS": prof, "INVESTMENT": inv}).to_csv(csv, index=False)

    rates = QuarterlyRates.load(csv)

    assert list(rates.df.columns) == [COL_PROFITS, COL_INVEST]
    assert len(rates) == 4                        # el primer trimestre cae por el diff
    np.testing.assert_allclose(rates.p, 100 * np.diff(np.log(prof)))
    np.testing.assert_allclose(rates.i, 100 * np.diff(np.log(inv)))
    assert rates.index[0] == idx[1]


# --------------------------------------------------------------- crisis_anchors
def test_anchor_search_respects_nber_window():
    # Índice 1947Q1–1951Q4: solo la recesión de 1949 tiene ventana no vacía.
    # Ventana de búsqueda: pico(1948-12-31)−2 trim = 1948-06-30 … valle(1949-12-31)+4 trim = 1950-12-31.
    idx = pd.date_range("1947-03-31", periods=20, freq="QE")
    i = np.zeros(20)
    i[list(idx).index(pd.Timestamp("1948-03-31"))] = -10.0   # mínimo global, FUERA de la ventana
    i[list(idx).index(pd.Timestamp("1949-09-30"))] = -5.0    # mínimo DENTRO de la ventana
    rates = make_rates(np.zeros(20), i)

    anchors = rates.crisis_anchors()

    assert len(anchors) == 1
    lbl, a, ts = anchors[0]
    assert lbl == "1949"
    assert ts == pd.Timestamp("1949-09-30")       # elige el mínimo de la ventana, no el global


def test_anchor_window_edges_are_inclusive():
    idx = pd.date_range("1947-03-31", periods=20, freq="QE")
    for edge in ("1948-06-30", "1950-12-31"):     # primer y último trimestre de la ventana
        i = np.zeros(20)
        i[list(idx).index(pd.Timestamp(edge))] = -5.0
        anchors = make_rates(np.zeros(20), i).crisis_anchors()
        assert anchors[0][2] == pd.Timestamp(edge)


# --------------------------------------------------------------- event_windows
def test_event_windows_drops_anchors_near_sample_edges():
    p = np.arange(20.0)
    rates = make_rates(p, p * 2)
    study = EventStudy(rates, h=2,
                       anchors=[("A", 1, None), ("B", 5, None), ("C", 18, None)])

    usable, P, I = study.event_windows()

    assert [lbl for lbl, _ in usable] == ["B"]    # A pisa el borde izq., C el derecho
    np.testing.assert_array_equal(P[0], p[3:8])   # ventana ±2 centrada en el ancla
    np.testing.assert_array_equal(I[0], p[3:8] * 2)
    assert P.shape == (1, 5)


# --------------------------------------------------------------- crisis_regimes
def test_crisis_regimes_partition_and_drop():
    rates = make_rates(np.zeros(40), np.zeros(40))
    study = EventStudy(rates, anchors=[("A", 10, None), ("B", 25, None)])

    seg_near, seg_exp, near = study.crisis_regimes()
    assert [list(s) for s in seg_near] == [list(range(4, 17)), list(range(19, 32))]
    assert [list(s) for s in seg_exp] == [list(range(0, 4)), list(range(17, 19)),
                                          list(range(32, 40))]
    assert near.sum() == 26                       # 2 ventanas de ±6 → 13 trim c/u

    # al excluir B, su ventana no va a NINGÚN régimen
    seg_near, seg_exp, _ = study.crisis_regimes(drop_labels=("B",))
    assert [list(s) for s in seg_near] == [list(range(4, 17))]
    assert [list(s) for s in seg_exp] == [list(range(0, 4)), list(range(17, 19)),
                                          list(range(32, 40))]


# --------------------------------------------------------------- segmentos y pares
def test_contiguous_segments():
    mask = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    segs = contiguous_segments(mask)
    assert [list(s) for s in segs] == [[0, 1], [3], [6, 7, 8]]
    assert contiguous_segments(np.zeros(5, bool)) == []


def test_pooled_corr_lag_does_not_cross_segment_borders():
    rng = np.random.default_rng(0)
    p, i = rng.normal(size=12), rng.normal(size=12)
    segs = [np.arange(0, 6), np.arange(6, 12)]
    k = -1

    c, n = pooled_corr_lag(p, i, segs, k)

    # pares armados a mano, sin cruzar el borde entre los segmentos
    P = np.concatenate([p[1:6], p[7:12]])
    I = np.concatenate([i[0:5], i[6:11]])
    assert n == len(P) == 10                      # Σ (len(seg) − |k|)
    np.testing.assert_allclose(c, np.corrcoef(P, I)[0, 1])

    # la versión ingenua (serie entera, cruzando el borde) daría otro número
    naive = np.corrcoef(p[1:], i[:-1])[0, 1]
    assert not np.isclose(c, naive)


def test_pooled_corr_lag_skips_segments_shorter_than_lag():
    rng = np.random.default_rng(1)
    p, i = rng.normal(size=10), rng.normal(size=10)
    segs = [np.arange(0, 3), np.arange(3, 10)]    # el primero (len 3) no alcanza para k=−4
    c, n = pooled_corr_lag(p, i, segs, -4)
    np.testing.assert_allclose(c, np.corrcoef(p[7:10], i[3:6])[0, 1])
    assert n == 3
