"""The three monitors drift-sentry runs against a model's production stream.

1. Input-drift monitor: PSI per feature per hour against a frozen reference
   window. Catches data drift and pipeline bugs within hours, needs no labels.
2. Score-drift monitor: PSI on the model's output distribution. Catches
   anything that changes what the model is saying, also label-free.
3. Output-quality SLO: precision of the model's flags on matured (48h-old)
   labels, rolling 24h. The only monitor that can catch concept drift, and it
   is structurally late by the label lag.

PSI rule of thumb: <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant.
"""
import numpy as np

from traces import REF_HOURS, LABEL_LAG_H, COUNTRIES

PSI_ALERT = 0.2
PSI_SUSTAIN_H = 3       # consecutive hours over threshold before alerting
QUALITY_SUSTAIN_H = 6


def psi(ref_frac, cur_frac, eps=1e-4):
    r = np.clip(ref_frac, eps, None)
    c = np.clip(cur_frac, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _hist_frac(values, edges):
    h, _ = np.histogram(values, bins=edges)
    return h / max(h.sum(), 1)


def _country_frac(idx):
    h = np.bincount(idx, minlength=len(COUNTRIES))
    return h / max(h.sum(), 1)


def psi_series(hours):
    """Hourly PSI for each monitored distribution vs the reference window."""
    amt_edges = np.linspace(0.0, 7.5, 16)
    age_edges = np.array([-0.5, 3, 7, 14, 30, 60, 90, 150, 250, 400, 10000])
    score_edges = np.linspace(0, 1, 21)

    ref = hours[:REF_HOURS]
    ref_amt = _hist_frac(np.concatenate([h.log_amt for h in ref]), amt_edges)
    ref_age = _hist_frac(np.concatenate([h.device_age for h in ref]), age_edges)
    ref_cty = _country_frac(np.concatenate([h.country_idx for h in ref]))
    ref_score = _hist_frac(np.concatenate([h.scores for h in ref]), score_edges)

    out = {"amount": [], "device_age": [], "country": [], "score": []}
    for h in hours:
        out["amount"].append(psi(ref_amt, _hist_frac(h.log_amt, amt_edges)))
        out["device_age"].append(psi(ref_age, _hist_frac(h.device_age, age_edges)))
        out["country"].append(psi(ref_cty, _country_frac(h.country_idx)))
        out["score"].append(psi(ref_score, _hist_frac(h.scores, score_edges)))
    return {k: np.array(v) for k, v in out.items()}


def flag_threshold(hours):
    """Operating point: flag the top ~1.5% of scores, set on the reference window."""
    ref_scores = np.concatenate([h.scores for h in hours[:REF_HOURS]])
    return float(np.quantile(ref_scores, 0.985))


def precision_series(hours, thr):
    """Rolling 24h precision of flags, computed only on matured labels.
    Value at hour t covers transactions from hours [t-48-24, t-48)."""
    n_hours = len(hours)
    flagged = np.zeros(n_hours)
    flagged_fraud = np.zeros(n_hours)
    for t, h in enumerate(hours):
        f = h.scores >= thr
        flagged[t] = f.sum()
        flagged_fraud[t] = (f & h.fraud).sum()

    prec = np.full(n_hours, np.nan)
    for t in range(n_hours):
        hi = t - LABEL_LAG_H          # newest matured hour (exclusive)
        lo = hi - 24
        if lo < 0:
            continue
        fl = flagged[lo:hi].sum()
        if fl > 0:
            prec[t] = flagged_fraud[lo:hi].sum() / fl
    return prec


def sustained_alerts(breach_mask, sustain):
    """Alert start times: first hour of each episode where the mask has been
    true for `sustain` consecutive hours."""
    alerts = []
    run = 0
    in_alert = False
    for t, b in enumerate(breach_mask):
        run = run + 1 if b else 0
        if run >= sustain and not in_alert:
            alerts.append(t)
            in_alert = True
        if not b:
            in_alert = False
    return alerts
