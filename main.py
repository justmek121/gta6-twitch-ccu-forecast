"""Forecast a game launch day's Twitch viewers, hour by hour, from hour 1.

Written for make.fun's gta6-twitch-ccu-forecast-01 bounty (program/v1,
Python 3.13, standard library only, no network).

Idea
----
Every case carries a library of comparable past launches, each with its
full hourly viewer curve from 14 days before its release day to 3 days
after. A launch's release-day shape, measured as the log ratio between
each later hour and its own hour 1, is not random: it depends on how big
hour 1 already was (a category already near Twitch's ceiling cannot
multiply tenfold by evening, a small one can) and on where hour 1 sits
against the launch's own pre-release peak (an hour 1 that is a trough
after an early-regions rush is followed by a strong climb when the US
unlocks and prime time arrives).

So for each forecast hour this program fits a small ridge regression
across the comparables in the case:

    log(viewers_h / viewers_1) ~ a_h + b_h * z(log viewers_1)
                                   + c_h * z(log(viewers_1 / pre_peak))

with the log ratios winsorized to [-3, 3] so a comparable whose hour 1
was a pre-launch trickle cannot drag the fit, the two features
standardized, and a ridge penalty of 10 on the slopes. The spread comes
from the fit's residuals: their interquartile range gives a per-hour
sigma, and the five quantiles are the prediction plus the standard
normal's z at each level times sigma times 0.85, a calibration factor
chosen by leave-one-out validation over the rolling-midnight launches
among the comparables. Forecasts are anchored on the case's own hour-1
print, so the same program runs unchanged on the practice launches and
on GTA VI's release day.

The program reads every *.in file in the input directory and writes one
<id>.out per case, deterministically: no randomness, no clock, no
iteration over unsorted containers.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RIDGE_LAMBDA = 10.0
SPREAD_SCALE = 0.85
WINSOR = 3.0
MIN_TRAIN = 8
BASELINE_SIGMA = 0.15

_NORM = NormalDist()


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clip(value, low, high):
    return low if value < low else high if value > high else value


def quantile(sorted_values, level):
    """A quantile of pre-sorted values, by linear interpolation."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * level
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def solve_linear(matrix, vector):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    size = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        if abs(pivot_value) < 1e-12:
            continue
        for r in range(size):
            if r != col and rows[r][col] != 0.0:
                factor = rows[r][col] / pivot_value
                for c in range(col, size + 1):
                    rows[r][c] -= factor * rows[col][c]
    return [
        rows[i][size] / rows[i][i] if abs(rows[i][i]) > 1e-12 else 0.0
        for i in range(size)
    ]


class Launch:
    """One comparable launch: its hour-1 level and its day's log ratios."""

    def __init__(self, name, hour1, pre_peak, log_ratios):
        self.name = name
        self.hour1 = hour1
        self.features = (
            math.log(hour1),
            math.log(hour1 / pre_peak) if pre_peak and pre_peak > 0 else 0.0,
        )
        self.log_ratios = log_ratios


def read_comparable(comp, game_name):
    """Turn one comparable record into a Launch, or None if unusable."""
    name = str(comp.get("name", ""))
    if game_name and game_name in name:
        return None  # the case's own category history is not a launch
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, ValueError):
        return None
    hourly = {}
    for point in comp.get("hourly", []):
        try:
            viewers = point["viewers"]
            when = parse_time(point["from"])
        except (KeyError, ValueError, TypeError):
            continue
        if viewers is not None and viewers > 0:
            hourly[when] = float(viewers)
    hour1 = hourly.get(day_start)
    if not hour1:
        return None
    window_start = day_start - timedelta(hours=12)
    pre_values = [v for t, v in hourly.items() if window_start <= t < day_start]
    pre_peak = max(pre_values) if pre_values else hour1
    log_ratios = {}
    for hour in range(2, 25):
        viewers = hourly.get(day_start + timedelta(hours=hour - 1))
        if viewers:
            log_ratios[hour] = clip(math.log(viewers / hour1), -WINSOR, WINSOR)
    if len(log_ratios) < 12:
        return None
    return Launch(name, hour1, pre_peak, log_ratios)


def target_features(case, hour1):
    """The case's own two features, matching the comparables' definitions."""
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, ValueError):
        day_start = None
    pre_peak = None
    if day_start is not None:
        window_start = day_start - timedelta(hours=12)
        values = []
        for point in case.get("history_hourly") or []:
            try:
                when = parse_time(point["from"])
                viewers = point["viewers"]
            except (KeyError, ValueError, TypeError):
                continue
            if viewers and window_start <= when < day_start:
                values.append(float(viewers))
        if not values:
            for point in case.get("history") or []:
                try:
                    when = parse_time(point["at"])
                    viewers = point["viewers"]
                except (KeyError, ValueError, TypeError):
                    continue
                if viewers and window_start <= when < day_start:
                    values.append(float(viewers))
        if values:
            pre_peak = max(values)
    if not pre_peak:
        pre_peak = hour1
    return (math.log(hour1), math.log(hour1 / pre_peak))


def forecast_hour(hour, launches, features, levels):
    """Predict one hour's quantiles from the comparable launches."""
    train = [launch for launch in launches if hour in launch.log_ratios]
    if len(train) < MIN_TRAIN:
        return None
    # Standardize the two features on this hour's training set.
    stats = []
    for idx in (0, 1):
        column = [launch.features[idx] for launch in train]
        mean = sum(column) / len(column)
        var = sum((v - mean) ** 2 for v in column) / len(column)
        stats.append((mean, math.sqrt(var) or 1.0))
    rows = [
        [1.0] + [(launch.features[i] - stats[i][0]) / stats[i][1] for i in (0, 1)]
        for launch in train
    ]
    target = [launch.log_ratios[hour] for launch in train]
    size = 3
    matrix = [
        [
            sum(row[i] * row[j] for row in rows)
            + (RIDGE_LAMBDA if i == j and i > 0 else 0.0)
            for j in range(size)
        ]
        for i in range(size)
    ]
    vector = [sum(row[i] * y for row, y in zip(rows, target)) for i in range(size)]
    beta = solve_linear(matrix, vector)
    point = [1.0] + [(features[i] - stats[i][0]) / stats[i][1] for i in (0, 1)]
    prediction = sum(beta[i] * point[i] for i in range(size))
    residuals = sorted(
        y - sum(beta[i] * row[i] for i in range(size))
        for row, y in zip(rows, target)
    )
    iqr = quantile(residuals, 0.75) - quantile(residuals, 0.25)
    sigma = max(iqr / 1.349, 0.05)
    return [
        prediction + _NORM.inv_cdf(level) * sigma * SPREAD_SCALE
        for level in levels
    ]


def hour1_print(case):
    """Hour 1's print: the known row, else the last history point's level."""
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is not None and known.get("viewers"):
        return max(1.0, float(known["viewers"]))
    history = case.get("history") or []
    if history and history[-1].get("viewers"):
        return max(1.0, float(history[-1]["viewers"]))
    raise ValueError("case has no hour-1 print")


def forecast_case(case):
    """Build the forecast object for one parsed case."""
    levels = [float(x) for x in case.get("levels", [0.05, 0.25, 0.5, 0.75, 0.95])]
    hour1 = hour1_print(case)
    game_name = str(case.get("game", {}).get("name", ""))
    launches = []
    for comp in case.get("comparables", []):
        launch = read_comparable(comp, game_name)
        if launch is not None:
            launches.append(launch)
    features = target_features(case, hour1)
    hours = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        hour = int(row["hour"])
        log_quantiles = forecast_hour(hour, launches, features, levels)
        if log_quantiles is None:
            spread = BASELINE_SIGMA * math.sqrt(hour - 1)
            log_quantiles = [_NORM.inv_cdf(level) * spread for level in levels]
        quantiles = []
        for log_q in log_quantiles:
            value = hour1 * math.exp(clip(log_q, -WINSOR, WINSOR))
            quantiles.append(clip(round(value, 2), 0.01, 1_000_000_000))
        quantiles.sort()
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
