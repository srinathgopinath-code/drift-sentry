"""drift-sentry: catch the model failures your uptime dashboard will never show.

Replays 14 days of production traffic for a fraud-scoring service whose HTTP
availability never drops, injects three silent failures (data drift, concept
drift, a feature-pipeline bug), and shows what three label-free-or-lagged
monitors see, and when.
"""
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traces import build_trace, EVENTS, REF_HOURS, LABEL_LAG_H, HOURS
from monitors import (psi_series, precision_series, flag_threshold,
                      sustained_alerts, PSI_ALERT, PSI_SUSTAIN_H, QUALITY_SUSTAIN_H)

EVIDENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
log = logging.getLogger("drift-sentry")


def setup_logging():
    os.makedirs(EVIDENCE, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(EVIDENCE, "run.log"), mode="w")])


def hfmt(t):
    return f"day {t / 24:.1f} (h{t})"


def chart_quality(availability, prec, slo, quality_alerts):
    t = np.arange(HOURS) / 24.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.plot(t, availability, color="tab:green", lw=1)
    ax1.set_ylim(99.9, 100.001)
    ax1.set_ylabel("HTTP availability (%)")
    ax1.set_title("What the uptime dashboard sees (top) vs what the model is doing (bottom)",
                  fontsize=10)
    ax2.plot(t, prec, color="tab:blue", lw=1.2, label="flag precision (24h, matured labels)")
    ax2.axhline(slo, color="red", ls="--", lw=1, label=f"quality SLO ({slo:.2f})")
    for name, et, _ in EVENTS:
        ax2.axvline(et / 24, color="gray", ls=":", lw=1)
        ax2.text(et / 24 + 0.05, 0.97, name, rotation=90, fontsize=7, va="top")
    for a in quality_alerts:
        ax2.axvline(a / 24, color="red", lw=1.5, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("day")
    ax2.set_ylabel("precision")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(EVIDENCE, "quality_vs_uptime.png"), dpi=110)
    plt.close(fig)


def chart_psi(psis, psi_alerts):
    t = np.arange(HOURS) / 24.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, series in psis.items():
        ax.plot(t, series, lw=1, label=f"PSI {name}")
    ax.axhline(PSI_ALERT, color="red", ls="--", lw=1, label="PSI alert threshold (0.2)")
    for name, et, _ in EVENTS:
        ax.axvline(et / 24, color="gray", ls=":", lw=1)
        ax.text(et / 24 + 0.05, ax.get_ylim()[1] * 0.95, name, rotation=90,
                fontsize=7, va="top")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("day")
    ax.set_ylabel("PSI vs reference (days 1-5)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(EVIDENCE, "psi_monitors.png"), dpi=110)
    plt.close(fig)


def main():
    setup_logging()
    log.info("drift-sentry: 14-day replay, reference window = days 1-5, "
             "label lag = %dh", LABEL_LAG_H)
    hours, availability = build_trace()
    log.info("trace built: %d hours, %d transactions, availability min=%.3f%% (never pages)",
             len(hours), sum(h.n for h in hours), availability.min())

    thr = flag_threshold(hours)
    prec = precision_series(hours, thr)
    ref_prec = np.nanmean(prec[:REF_HOURS + LABEL_LAG_H])
    slo = round(ref_prec - 0.10, 2)   # SLO derived from known-good baseline
    log.info("operating point: flag score>=%.3f; reference precision=%.2f -> quality SLO=%.2f",
             thr, ref_prec, slo)

    psis = psi_series(hours)
    psi_alerts = {name: sustained_alerts(series > PSI_ALERT, PSI_SUSTAIN_H)
                  for name, series in psis.items()}
    quality_alerts = sustained_alerts(np.nan_to_num(prec, nan=1.0) < slo,
                                      QUALITY_SUSTAIN_H)

    for name, alerts in psi_alerts.items():
        log.info("PSI monitor [%s]: alerts at %s", name,
                 [hfmt(a) for a in alerts] or "none")
    log.info("quality SLO monitor: alerts at %s",
             [hfmt(a) for a in quality_alerts] or "none")

    # ---- match alerts to injected events ----
    lines = ["# drift-sentry report", "",
             f"14-day replay. HTTP availability never dropped below "
             f"{availability.min():.3f}%: **zero uptime alerts**. "
             f"Flag threshold {thr:.3f} (top 1.5% of reference scores); "
             f"quality SLO precision >= {slo:.2f}.", "",
             "| injected event | when | caught by | detection | lag |",
             "|---|---|---|---|---|"]
    all_alerts = ([("PSI:" + n, a) for n, al in psi_alerts.items() for a in al]
                  + [("quality-SLO", a) for a in quality_alerts])
    all_alerts.sort(key=lambda x: x[1])
    next_event_start = {name: (et, HOURS if i + 1 >= len(EVENTS) else EVENTS[i + 1][1])
                        for i, (name, et, _) in enumerate(EVENTS)}
    for name, et, desc in EVENTS:
        lo, hi = next_event_start[name]
        matched = [(m, a) for m, a in all_alerts if lo <= a < hi]
        if matched:
            m, a = matched[0]
            extra = "" if len(matched) == 1 else f" (+{len(matched) - 1} more)"
            lines.append(f"| {name}: {desc} | {hfmt(et)} | {m}{extra} | {hfmt(a)} "
                         f"| {a - et}h |")
            log.info("EVENT %-13s detected by %-13s after %3dh %s",
                     name, m, a - et, extra)
        else:
            lines.append(f"| {name}: {desc} | {hfmt(et)} | NOTHING | - | MISSED |")
            log.info("EVENT %-13s MISSED", name)

    lines += ["", "## The point", "",
              "The uptime line was green for all 336 hours. Every one of these "
              "failures was invisible to availability monitoring and would have "
              "surfaced as slowly-eroding business metrics weeks later. "
              "Label-free monitors (PSI on inputs and scores) caught the "
              "pipeline bug and the market shift within hours; only the "
              "label-based quality SLO could catch the concept drift, and it "
              "is structurally late by the 48h label lag. You need both."]
    with open(os.path.join(EVIDENCE, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    chart_quality(availability, prec, slo, quality_alerts)
    chart_psi(psis, psi_alerts)
    log.info("wrote evidence/report.md, quality_vs_uptime.png, psi_monitors.png")
    log.info("done")


if __name__ == "__main__":
    main()
