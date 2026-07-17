"""Synthetic production trace for a fraud-scoring model, with injected silent failures.

14 days of hourly traffic through a deployed fraud model. The service never
returns an error: HTTP availability stays ~99.99% the whole time. Three silent
failures are injected, none of which any uptime monitor can see:

  day 6:  DATA DRIFT     - launch in a new market (MX) shifts the input mix
  day 9:  CONCEPT DRIFT  - fraud pattern flips to account-takeover (small
                           amounts, aged devices); the frozen model was trained
                           on the old pattern and quietly stops catching fraud
  day 12: PIPELINE BUG   - upstream schema change zeroes the device_age
                           feature for all traffic

Ground-truth labels (chargebacks) arrive with a 48h lag, as in real life.
"""
import numpy as np

HOURS = 14 * 24
REF_HOURS = 5 * 24          # days 1-5: reference window, known-good
DATA_DRIFT_T = 6 * 24
CONCEPT_DRIFT_T = 9 * 24
PIPELINE_BUG_T = 12 * 24
LABEL_LAG_H = 48
SEED = 20260716

EVENTS = [
    ("data_drift", DATA_DRIFT_T, "MX market launch shifts input mix (25% new traffic)"),
    ("concept_drift", CONCEPT_DRIFT_T, "fraud flips to account-takeover pattern"),
    ("pipeline_bug", PIPELINE_BUG_T, "schema change zeroes device_age for all traffic"),
]

COUNTRIES = ["US", "GB", "DE", "IN", "BR", "SG", "MX"]
BASE_MIX = np.array([0.40, 0.15, 0.12, 0.12, 0.11, 0.10, 0.00])
DRIFT_MIX = np.array([0.30, 0.11, 0.09, 0.09, 0.08, 0.08, 0.25])
COUNTRY_RISK = np.array([0.20, 0.20, 0.10, 0.50, 0.60, 0.30, 0.45])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _deployed_model(log_amt, c_risk, device_age, rng):
    """The model actually serving traffic. Weights frozen at train time
    (pre-drift world): fraud = large amounts, risky country, brand-new device."""
    new_dev = (device_age < 7).astype(float)
    z = (-5.2 + 1.8 * (log_amt - 3.5) + 3.0 * c_risk + 2.8 * new_dev
         + rng.normal(0, 0.12, size=len(log_amt)))
    return _sigmoid(z)


def _true_fraud_prob(log_amt, c_risk, device_age, concept_shifted):
    """The world's actual fraud process. Shifts at CONCEPT_DRIFT_T."""
    new_dev = (device_age < 7).astype(float)
    old_dev = (device_age > 90).astype(float)
    if not concept_shifted:
        z = -6.2 + 1.9 * (log_amt - 3.5) + 3.2 * c_risk + 3.0 * new_dev
    else:
        # account-takeover wave: small test transactions from long-lived devices
        z = -6.8 - 1.3 * (log_amt - 3.5) + 1.5 * c_risk + 2.8 * old_dev
    return _sigmoid(z)


class HourBatch:
    __slots__ = ("scores", "fraud", "log_amt", "country_idx", "device_age", "n")

    def __init__(self, scores, fraud, log_amt, country_idx, device_age):
        self.scores = scores
        self.fraud = fraud
        self.log_amt = log_amt
        self.country_idx = country_idx
        self.device_age = device_age
        self.n = len(scores)


def build_trace():
    rng = np.random.default_rng(SEED)
    hours = []
    for t in range(HOURS):
        # diurnal volume, 1500-6000 tx/hour
        hod = t % 24
        n = int(rng.poisson(1500 + 4500 * 0.5 * (1 + np.sin(2 * np.pi * (hod - 9) / 24))))

        mix = DRIFT_MIX if t >= DATA_DRIFT_T else BASE_MIX
        country_idx = rng.choice(len(COUNTRIES), size=n, p=mix)
        c_risk = COUNTRY_RISK[country_idx]

        # MX cohort spends a little less
        log_amt = rng.normal(3.5, 1.0, size=n)
        log_amt[country_idx == 6] = rng.normal(3.1, 0.8, size=int((country_idx == 6).sum()))

        true_device_age = rng.gamma(2.0, 60.0, size=n)
        # what the model actually receives (bug zeroes the feature)
        seen_device_age = np.zeros(n) if t >= PIPELINE_BUG_T else true_device_age

        scores = _deployed_model(log_amt, c_risk, seen_device_age, rng)
        p_fraud = _true_fraud_prob(log_amt, c_risk, true_device_age,
                                   concept_shifted=(t >= CONCEPT_DRIFT_T))
        fraud = rng.random(n) < p_fraud

        hours.append(HourBatch(scores, fraud, log_amt, country_idx, seen_device_age))

    # the only thing an uptime monitor sees: a flat, healthy availability line
    availability = 99.99 + 0.009 * np.clip(rng.normal(0, 0.3, HOURS), -1, 0.9)
    return hours, availability
