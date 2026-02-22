# КП №5: Дослідження сигналу Н4-6 (Django)

This repository contains a Django app for task 5:
researching short-term instability of the universal calibrator **Н4-6** output signal as expanded uncertainty **U0.95(t)** versus observation time.

## 1. What this app does

The app takes measured signal values, converts them to relative deviation in ppm, builds uncertainty bounds for several observation windows, compares 3 competing model branches, chooses the best branch for the dataset, and shows results in an interactive web UI.

## 2. Input data

- File: `data/raw_signal.txt`
- Samples: `120`
- Sampling period (for default duration 60 min): `0.5 min`

Parser supports values like:
- `1,049990700E+00`
- decimal comma + scientific notation

## 3. Processing pipeline

1. Parse raw text into float values.
2. Compute basic statistics for raw signal.
3. Compute relative deviation in ppm:
   - `delta_i = (x_i - mean_x) / mean_x * 1e6`
4. Split data into windows: `1, 2, 5, 10, 30, 60` minutes.
5. For each window, split into equal segments.
6. For each segment, calculate bounds using all three model branches.
7. Average segment bounds to get `U_low(t)` and `U_up(t)` for each window.
8. Rank model branches by a competition score.
9. Fit cubic polynomials (degree 3) for lower and upper bounds.

## 4. Model branches (the competing "survival" branches)

### Branch A: Empirical quantiles
- Idea: no normality assumption.
- Strong side: fully data-driven.
- Critique: can be unstable on short/small segments.

### Branch B: Normal model
- Formula: `U = mu ± k*sigma` for chosen confidence `p`.
- Strong side: simple, interpretable, often stable.
- Critique: may be too optimistic for heavy tails/outliers.

### Branch C: Robust MAD model
- Formula: `U = median ± k*(1.4826*MAD)`.
- Strong side: resistant to outliers.
- Critique: with small segments can misestimate tails.

## 5. Branch competition and score

Each branch is evaluated by:
- coverage error (difference from target confidence `p`),
- mean interval width,
- width stability across segments.

Normalized weighted score:
- `score = 0.55*coverage_norm + 0.30*width_norm + 0.15*stability_norm`
- lower score = better branch.

## 6. Default result on current dataset (`p=0.95`)

Main stats:
- count: `120`
- mean: `1.0499939258333333`
- std: `2.322722823861736e-06`
- skewness: `0.22250004384329683`

Ranking:
1. **Branch B (normal)**, score `0.3000`
2. Branch C (robust), score `0.3903`
3. Branch A (quantile), score `0.7000`

Winner (Branch B) cubic models:
- `U_low(t) = 0.0000*t^3 - 0.0007*t^2 - 0.0453*t - 0.2434`
- `U_up(t)  = -0.0000*t^3 + 0.0007*t^2 + 0.0453*t + 0.2434`

(`t` in minutes, `U` in ppm)

## 7. Web interface features

- Controls:
  - confidence level `p`,
  - total observation duration (minutes),
  - window list,
  - histogram window selector.
- Buttons:
  - recalculate,
  - reset,
  - export CSV,
  - print.
- Charts:
  - raw signal `x(t)`,
  - relative deviation `delta(t)` in ppm,
  - `U0.95(t)` comparison for all branches,
  - histogram for selected window,
  - model score chart.

## 8. Project structure (important files)

- `analysis/services.py` - all analytics math and model competition.
- `analysis/views.py` - request parsing, context, CSV export.
- `analysis/templates/analysis/index.html` - GUI + Chart.js.
- `analysis/static/analysis/style.css` - styles.
- `analysis/urls.py` - app routes.
- `calibrator_lab/settings.py` - Django settings.

## 9. Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Run tests:

```bash
python manage.py test
```

## 10. Defense plan (what to say to professor)

### Step A: problem statement (30-45 sec)
"I implemented task 5 for calibrator Н4-6: estimate short-term instability as expanded uncertainty U0.95(t) versus observation time."

### Step B: data and preprocessing (45 sec)
"I used the provided 120 experimental samples, parsed scientific notation with decimal comma, converted all results to relative ppm."

### Step C: methodology (1.5-2 min)
"I implemented 3 competing branches: quantile, normal, robust MAD. For each observation window (1,2,5,10,30,60 min), I segmented the data, computed lower/upper bounds, averaged bounds, then ranked models by coverage, width, and stability."

### Step D: results (1 min)
"For this dataset, normal model won with the best integrated score. Then I fitted cubic polynomials for lower and upper uncertainty bounds to get analytical U(t) formulas."

### Step E: live demo (2 min)
Show in UI:
1. Change `p` from `0.95` to `0.90` and explain narrower intervals.
2. Show ranking table and branch critique.
3. Export CSV and print view.

### Step F: limitations (30 sec)
"The winner depends on dataset behavior. If tails/outliers change, robust or quantile branch can become better. Score weights are explicit and adjustable."

## 11. Typical professor questions and concise answers

Q: Why ppm?
A: It normalizes deviations by mean level, so uncertainty becomes scale-independent and comparable.

Q: Why these exact windows?
A: They follow the assignment methodology: 1, 2, 5, 10, 30, 60 minutes.

Q: Why cubic polynomial?
A: It gives smooth analytical approximation of measured `U(t)` points and is standard for this assignment style.

Q: Why not only one model?
A: Single-model choice can be biased. Competition provides evidence-based selection.

Q: What if distribution is non-normal?
A: That is why quantile and robust branches are implemented and compared.

## 12. Fast edits before defense

- Change default windows: `analysis/services.py` (`DEFAULT_WINDOWS`).
- Change score weights: `analysis/services.py` (inside `model_competition`).
- Change formulas or branch logic: `analysis/services.py`.
- Change UI text/layout: `analysis/templates/analysis/index.html`.
