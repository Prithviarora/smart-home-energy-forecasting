# Smart Home Energy Forecasting & Solar Load Shifting

Hour-ahead forecasting of household electricity demand and rooftop solar
generation, used to evaluate whether shifting flexible appliance loads into
peak generation windows meaningfully reduces grid import.

**[Open the notebook →](Smart_Home_Energy_Forecasting.ipynb)**

---

## Data

503,910 minute-resolution smart-meter readings from a single Massachusetts
household, January–December 2016, with 14 submetered circuits and co-located
weather observations. Aggregated to 8,399 hourly records.

Source: [Smart Home Dataset with Weather Information](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information)
(Kaggle, ~130 MB). Not included in this repository — download `HomeC.csv` and
upload it when prompted by the first cell.

## Findings

**The dataset's timestamp column is corrupted.** The `time` field increments by
one second per row while the meter samples once per minute, so it spans 5.8
days rather than 349. Parsed directly, it produces null time-of-day features —
meaning any analysis that uses it without checking has no working hour or
weekday signal at all. The index was reconstructed from the known sampling
cadence, after which the daily profile resolves correctly.

**Peak demand and peak generation do not coincide.** Demand peaks at 18:00
(1.29 kW) when solar output has fallen to zero; generation peaks at 11:00 when
the house draws its daily minimum. This mismatch is the entire case for load
shifting on this system.

**Leakage severity scales with sampling resolution.** A controlled 2×2 ablation
over split strategy and feature construction:

| R², `Fridge [kW]` | Random split | Chronological split |
|---|---|---|
| Concurrent features, 1-min | **0.922** | 0.163 |
| Concurrent features, hourly | 0.633 | 0.429 |
| Lagged features, hourly | 0.530 | 0.527 |

Shuffling minute-level data inflates R² by 0.76; at hourly resolution the same
error costs 0.20; with correctly lagged features the split barely matters
(0.004). The two faults interact rather than add — with concurrent features the
model has no legitimate temporal information, so a shuffled split lets it
interpolate between adjacent rows.

**Above a 2× array, the binding constraint is flexible load, not generation.**
Grid-import reduction across PV array sizes: 0.60% → 1.77% → 3.00% → 3.84% at
1×, 2×, 4×, 8×. Doubling from 1× to 2× nearly triples the benefit; 4× to 8×
adds under a point. Beyond that point every shiftable load has already been
moved into daylight and additional panels have nothing left to absorb.

## Model performance

Chronological split, 1,647 test hours (Oct–Dec 2016).

| Target | Model | MAE | RMSE | R² | Skill vs. persistence |
|---|---|---|---|---|---|
| Demand | Random Forest | 0.211 kW | 0.318 | 0.565 | +11.5% |
| Demand | HistGradientBoosting | 0.213 kW | 0.325 | 0.546 | +10.9% |
| Demand | Ridge | 0.222 kW | 0.332 | 0.526 | +7.1% |
| Solar | Random Forest | 0.014 kW | 0.032 | 0.892 | +31.0% |

Benchmarked against persistence, seasonal-naive, and hour-of-week climatology.
Persistence is the strongest baseline for demand (MAE 0.239); seasonal-naive
scores a negative R², indicating this household has little day-to-day routine.

The gap between the two skill scores reflects the difference between
forecasting a deterministic physical process and forecasting human behaviour.

## Method

- Every feature is knowable strictly before the predicted hour: history lagged
  ≥1h, rolling statistics computed on lagged values, calendar features
  cyclically encoded, weather treated as a forecast input.
- No submeter readings enter the demand model — individual circuits at time *t*
  sum into total consumption at time *t*.
- Train/test split chronologically, never shuffled.
- Scheduling decisions use forecast solar; savings scored against actual solar.
- Only discretionary batch loads are treated as shiftable. A fridge compressor
  and a thermostat-driven furnace cannot be scheduled.

## Limitations

Permutation importance shows the demand forecast is dominated by the 1-hour lag
(roughly 6× the next feature); the 11 weather features contribute negligibly.
The model is largely "recent state plus time of day," and the 11.5% skill score
should be read in that light. A longer forecast horizon would likely widen the
margin over persistence, which degrades quickly beyond one hour. Results come
from a single household and do not generalise without further testing.

## Stack

Python · pandas · scikit-learn · matplotlib
