# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/justmek121/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/justmek121/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for a game launch day on Twitch. Every case the grader runs carries a library of comparable past launches with their full hourly viewer curves, and the program fits, separately for each forecast hour, a small ridge regression across those comparables: the log ratio between that hour and the launch's own hour 1, explained by how big hour 1 already was and by where hour 1 sits against the launch's pre-release peak. A category already near Twitch's ceiling cannot multiply tenfold by evening; an hour 1 that is a trough after an early-regions rush usually climbs hard when the US unlocks and prime time arrives, and the regression is where those two effects are priced. Log ratios are winsorized so a comparable whose hour 1 was a pre-launch trickle cannot drag the fit, and the five quantiles come from the fit's residual spread, calibrated by leave-one-out validation over the rolling-midnight launches among the comparables. Pure Python standard library, deterministic, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9c64jd63&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
