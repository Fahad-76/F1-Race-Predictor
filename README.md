# F1-Race-Predictor

A machine learning pipeline that predicts F1 race winners, podium finishers, and top-10 results using Random Forest classifiers trained on historical session and qualifying data sourced from the FastF1 API — achieving a *ROC-AUC of 0.974* on the 2025 season.

## Overview

Built using  a Random Forest classifier pipeline trained on 7 seasons of F1 data (2018–2024) that predicts race winners, podium finishers, and top-10 finishers with high accuracy by using relative pace deltas, qualifying position, and Monte Carlo simulation to model driver competition as a dependent probability problem.

The core insight is that raw lap times are meaningless across seasons and circuits — a 1:21 at Australia in 2018 is equivalent to a 1:15 in 2024 due to car development. By converting all times to **delta-to-session-leader**, the model learns driver performance relative to the field rather than absolute lap times, making features comparable across all seasons and tracks.

## Results

Evaluated on the full 2025 season (17 races) as a blind test set.

| Model | ROC-AUC | Recall | Missed |
|---|---|---|---|
| Winner | **0.974** | 94% | 1/17 races |
| Podium | **0.960** | 94% | 3/51 podiums |

**Winner top-pick accuracy: 10/17 races (59%)** — correctly identifying the race winner as the single highest-probability driver more than half the time from pre-race data alone.

---
ml-f1-pipeline/
├── data/
│   ├── raw/
│   │   ├── 2018/
│   │   │   ├── sessions.csv        # FP1/FP2/FP3/Q best lap times
│   │   │   └── race_results.csv    # finishing positions + grid
│   │   ├── 2019/
│   │   │   └── ...
│   │   └── {season}/               # one folder per season
│   └── processed/
│       └── f1_dataset.csv          # merged, feature-engineered training set
│
├── models/
│   ├── rf_winner.pkl               # trained winner classifier
│   └── rf_podium.pkl               # trained podium classifier
│
├── predictions/
│   └── 2026_round01.csv            # output predictions per race
│
├── reports/
│   ├── eval_winner.png             # ROC curve + feature importance plots
│   └── eval_podium.png
│
└── src/
    ├── data/
    │   ├── fetch_fastf1.py         # scrapes FastF1 API by season
    │   └── build_dataset.py        # builds training CSV from raw data
    └── models/
        ├── train.py                # trains Random Forest classifiers
        ├── evaluate.py             # evaluates on held-out test season
        └── predict.py              # live race prediction + Monte Carlo
        
## How It Works

### 1. Data Collection (`fetch_fastf1.py`)

Scrapes the [FastF1 API] for every race weekend in a given season. Run one season at a time by changing a single variable:

```python
TARGET_SEASON = 2024  # ← change this each run
```

For each round it fetches:
- **FP1, FP2, FP3** — best lap time per driver per session
- **Qualifying** — best lap time + grid position
- **Race results** — finishing position, constructor, points, status (DNF etc.)

Data is saved to `data/raw/{season}/` immediately after each season so progress is never lost if a run is interrupted mid-season.


### 2. Feature Engineering (`build_dataset.py`)

Merges all season folders into a single training dataset and engineers the core features:

**Delta-to-leader normalisation:**
```
fp1_delta   = driver FP1 time  − fastest FP1 time that weekend
fp2_delta   = driver FP2 time  − fastest FP2 time that weekend
fp3_delta   = driver FP3 time  − fastest FP3 time that weekend
quali_delta = driver quali time − pole position time
```

A delta of `0.000` means the driver led that session. A delta of `1.500` means they were 1.5 seconds off the pace. This normalisation makes times comparable across:
- Different seasons (6+ seconds of car development 2018→2024)
- Different circuits (Monaco 1:10 vs Spa 1:41)
- Different conditions (wet vs dry sessions)

**Grid penalty detection:**
```
quali_rank   = expected grid position based on quali_delta
grid_penalty = GridPosition − quali_rank
```
A positive `grid_penalty` means the driver received a penalty and dropped back. These drivers often have fresher power unit components and can recover through the field — a genuinely predictive signal.

**Target labels created:**
- `Winner` — finished P1
- `Podium` — finished P1, P2, or P3
- `Top10`  — finished inside the points

**Missing data handling:** Drivers who skip practice sessions or sprint weekends that don't have a third practice session have their times filled with the per-event median rather than dropping the entire row.

---

### 3. Model Training (`train.py`)

Two separate Random Forest classifiers — one for winner prediction, one for podium prediction. They are trained independently because their class imbalance rates differ (~5% winners vs ~15% podium finishers) and require separate tuning.

**Temporal split** — the most critical design decision. Instead of a random train/test split, the most recent season is always held out as the test set:

A random split would allow races from 2025 to leak into the training set, producing inflated accuracy scores that wouldn't reflect real-world performance.

**Model configuration:**
```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,    # prevents overfitting on rare podium/win events
    class_weight='balanced', # compensates for class imbalance
    n_jobs=-1
)
```

An `sklearn` Pipeline wraps an imputer and the classifier together so missing values are handled consistently at both train and predict time.

---

### 4. Evaluation (`evaluate.py`)

Evaluates both models on the held-out test season with metrics appropriate for a binary classification problem with class imbalance:

- **ROC-AUC** — primary metric. Measures how well the model ranks true positives above negatives regardless of threshold. Much more meaningful than accuracy for rare events like race wins
- **Classification report** — precision, recall, F1 per class
- **Confusion matrix** — with human-readable TP/FP/FN/TN breakdown
- **Per-race breakdown** — for each round: actual result vs predicted vs top-probability driver
- **Feature importance check** — flags if `GridPosition` is still dominating over `quali_delta`, which would suggest more training data is needed

Plots saved to `reports/`:
- ROC curve
- Feature importance bar chart
- Predicted probability distribution (positive vs negative class separation)

### 5. Live Race Prediction (`predict.py`)

Fetches live weekend data from FastF1 and runs predictions before the race starts. Change only one variable each weekend:

```python
SEASON = 2026
ROUND  = 1   # ← only this needs updating
```

**Feature pipeline mirrors training exactly:**
1. Fetch FP1/FP2/FP3/Q best lap times
2. Fetch qualifying grid positions
3. Compute delta-to-leader for all sessions
4. Compute grid penalty (qualifying rank vs actual grid slot)
5. Resolve TrackID from the training dataset's category encoding

**Monte Carlo simulation (1,000 races):**

Raw model probabilities are independent — the model scores each driver separately without knowing they are competing against each other. This is corrected with a dependent sampling simulation:

```
Each simulation:
  P1 → sample winner from normalised win probabilities
  P2 → remove winner, sample from remaining podium probabilities
  P3 → remove P1+P2, sample P3 from remaining
  P4–P20 → uniform shuffle of remaining drivers
```

Normalising probabilities so they sum to 1 across the field before each draw means one driver winning mathematically reduces all others' chances — correctly modelling the competitive dependency of a race.

After 1,000 simulations each driver has:
- `Win%` — percentage of simulations they won
- `Podium%` — percentage of simulations they finished top 3
- `Avg Finish` — average finishing position across all simulations



Example output is included in this repository in the predicitons folder.

### Why delta-to-leader instead of raw times?

| Problem with raw times | How deltas fix it |
|---|---|
| 2018 vs 2024 times differ by 6+ seconds | Delta is always relative to that weekend |
| Monaco lap ≈ 1:10, Spa lap ≈ 1:41 | Delta normalises across all circuits |
| Wet sessions produce slower times | A 0.3s delta means the same regardless |
| Car development inflates older seasons | 0.000 always means "led the session" |

### Why a classifier not a regressor?

Predicting finishing position (1–20) as a regression problem is less useful than predicting the probability of a binary outcome (won: yes/no). Probabilities can be ranked, compared across drivers, and fed into the Monte Carlo simulation. A regressor's position predictions also suffer from the artificial precision problem — predicting "3.7th place" is not meaningful.

### Why temporal not random train/test split?

F1 data is time-series. A random split with `random_state=42` would allow a 2024 race to train on 2025 data, producing inflated accuracy metrics that would not hold in production. The temporal split ensures evaluation always reflects real-world conditions: train on the past, test on the future.

### Why Monte Carlo for predictions?

The model assigns independent probabilities per driver. Without simulation, you could have VER at 40% win probability and NOR at 38% — those can't both simultaneously be true. Normalising and sampling 1,000 times forces the probabilities to be mutually exclusive and collectively exhaustive, which is the correct probabilistic framing for a race.

---

## Usage

### Install dependencies

```bash
pip install fastf1 pandas scikit-learn joblib matplotlib
```

### Step 1 — Fetch data (one season at a time)

```python
# In src/data/fetch_fastf1.py
TARGET_SEASON = 2022
```
```bash
python src/data/fetch_fastf1.py
```

Repeat for each season (2018–2025 recommended).

### Step 2 — Build training dataset

```bash
python src/data/build_dataset.py
```

### Step 3 — Train models

```bash
python src/models/train.py
```

### Step 4 — Evaluate

```bash
python src/models/evaluate.py
```

### Step 5 — Predict a race

```python
# In src/models/predict.py
SEASON = 2026
ROUND  = 1
```
```bash
python src/models/predict.py
```

### Retraining as the season progresses

Once enough 2026 races have completed, I will work to retrain the model and add more parameters such as driver form and constructor points.

---

## Limitations

**What the model cannot predict:**
- Safety car periods and their strategic impact
- Mechanical failures and DNFs
- Weather changes during the race
- First-lap incidents that reshuffle the order
- Drivers on penalty-fuelled power unit upgrades
- New drivers or constructors with no historical data (e.g. 2026 regulation changes)

**GridPosition still dominant:** At 0.37 feature importance, grid position remains the strongest single predictor. This is directionally correct — starting position is the strongest real-world predictor of race finish — but it means the model leans heavily on qualifying results. Adding rolling driver/constructor form features would reduce this dependency.

**New circuits:** Tracks introduced after the training cutoff receive a `TrackID` of `-1`. Predictions at these venues rely entirely on session times and grid position with no track-specific pattern recognition.

---

## Requirements

```
Python 3.9+
fastf1>=3.0
pandas>=1.5
scikit-learn>=1.2
joblib
matplotlib
numpy
```

---

## Data Source

All data sourced from the [FastF1](https://docs.fastf1.dev/) Python library which provides access to official F1 timing data, telemetry, and session results. Reliable data available from the 2018 season onward.

