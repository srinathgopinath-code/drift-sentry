# drift-sentry

Catch the model failures your uptime dashboard will never show you.

## The problem

Traditional reliability monitoring assumes failures are loud: errors, timeouts,
5xx spikes. Production ML systems break that contract. A fraud model whose
precision has been decaying for weeks, a feature pipeline quietly zeroed by an
upstream schema change, a market launch that shifts the input population — all
of these happen while HTTP availability reads 99.99% and every dashboard stays
green. Nothing in the standard SRE toolbox is looking for them.

## What this does

`drift-sentry` demonstrates, end to end, the minimum monitoring set that makes
silent model failure visible, and measures how fast each monitor catches what:

- **Input-drift monitor** — PSI (population stability index) per feature per
  hour against a frozen known-good reference window. Label-free.
- **Score-drift monitor** — PSI on the model's output distribution. Label-free.
- **Output-quality SLO** — rolling precision of the model's flags computed on
  matured ground-truth labels (48h chargeback lag), with the SLO derived from
  the reference baseline.

It replays 14 days of realistic traffic (1.26M transactions) through a frozen
fraud model and injects three silent failures: a **data drift** (new market =
25% of traffic), a **concept drift** (fraud flips to an account-takeover
pattern the model was never trained on), and a **feature-pipeline bug**
(schema change zeroes `device_age` for all traffic).

## Results (this run)

HTTP availability never left 99.98%: zero uptime alerts across all 336 hours.

| injected failure | caught by | detection lag |
|---|---|---|
| data drift (day 6) | PSI on country mix | 2h |
| concept drift (day 9) | quality SLO only | 64h (48h of it = label lag) |
| pipeline bug (day 12) | PSI on device_age + score | 2h |

The asymmetry is the finding: label-free drift monitors catch pipeline bugs
and population shifts within hours, but are structurally blind to concept
drift (the inputs never changed — the world did). Only the label-based quality
SLO catches that, and it can never be faster than your label lag. You need
both, and you need to know your label lag, because it is a floor on your
time-to-detect. Full scorecard in `evidence/report.md`, charts and raw run
log alongside it.

## Run it

```bash
pip install -r requirements.txt
python run.py
```

Deterministic (fixed seed); reproduces the exact numbers above in ~2 seconds.

## Adapt it

- Point `precision_series` / `psi_series` in `monitors.py` at your own
  prediction log (score, features, delayed label per event) instead of the
  synthetic trace.
- Set `LABEL_LAG_H` to your real label latency; the report will show you what
  detection floor you have accepted.
- Tune `PSI_ALERT` / sustain windows to your traffic volume.

## Layout

```
traces.py     14-day synthetic production stream with 3 injected silent failures
monitors.py   PSI drift monitors + matured-label quality SLO + alert logic
run.py        runs everything, writes evidence/ (report, charts, log)
evidence/     captured output of a real run
```
