# Appliance Energy Prediction
### Multivariate Time-Series Deep Learning - Take-Home Assessment

**Author:** [Your Name]
**Date:** 2026-07-04
**Assessment by:** Intrinsic Tech

---

## Project Overview

This project predicts household appliance energy consumption (Wh) using multivariate
time-series deep learning. The dataset contains environmental sensor readings from a
low-energy house recorded at 10-minute intervals across January to May 2016.

Seven models are implemented and compared:

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Baseline |
| LSTM | Meets assessment requirement |
| GRU | Meets assessment requirement |
| CNN-LSTM | Meets assessment requirement |
| TCN (Temporal Convolutional Network) | Initiative addition |
| CNN-LSTM with Attention | Primary model - initiative addition |

An eighth model - Linear Regression + GRU stacked on its residuals - was
added during optimization (notebook 05) after discovering it was the only
approach that beat the Linear Regression baseline outright. See
"Further Optimization & Ceiling Analysis" below.

---

## Repository Structure
```
energy-prediction/
├── data/
│   ├── raw/                        - Original dataset (energy_data_set.csv)
│   └── processed/                  - Cleaned and feature-engineered outputs,
│                                      selected_features.txt
├── notebooks/
│   ├── 01_EDA.ipynb                - Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_Deep_Learning.ipynb
│   └── 05_Optimization_Evaluation.ipynb  - Optimization, ablations, ceiling analysis
├── src/
│   ├── data_preprocessing.py       - Loading, cleaning, scaling functions
│   ├── feature_engineering.py      - Feature creation and selection functions
│   ├── model.py                    - All model architecture definitions
│   ├── train.py                    - Training loop and evaluation pipeline
│   └── run_xgboost_subprocess.py   - XGBoost, run out-of-process (see Known
│                                      Environment Issue below)
├── models/                         - Saved trained model files (.h5 / .pt / .pkl)
├── reports/
│   ├── figures/                    - All generated plots
│   ├── final_results_table.csv     - Master comparison, every model/variant
│   ├── extended_experiments_results.csv, ablation_results.csv,
│   │   optimization_results.csv, baseline_results.csv
│   └── report.pdf                  - Final assessment report (in progress)
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Environment Setup

### Option 1 - Local with venv (Recommended)

```bash
# Clone the repository
git clone https://github.com/Muaadh-Nazly/energy-prediction.git
cd energy-prediction

# Create and activate virtual environment
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Option 2 - Google Colab

1. Upload `energy_data_set.csv` to your Colab session or mount Google Drive
2. Install the one extra dependency:
```python
   !pip install statsmodels
```
3. Adjust the `DATA_PATH` variable at the top of each notebook to your file location
4. Enable GPU: Runtime > Change runtime type > T4 GPU

---

## Running the Project

Run notebooks in order. Each notebook saves its output to `data/processed/`
for the next notebook to load.

01_EDA.ipynb                - EDA, outlier detection, ACF/PACF analysis
02_Feature_Engineering.ipynb - Time features, rolling windows, lag features, feature selection
03_Baseline_Models.ipynb    - Linear Regression and Random Forest benchmarks
04_Deep_Learning.ipynb      - LSTM, GRU, CNN-LSTM, TCN, CNN-LSTM+Attention
05_Optimization_Evaluation.ipynb - Hyperparameter tuning, ablations, stacking,
                               ARIMA/XGBoost comparisons, final evaluation

`src/` is a shared library, not a standalone CLI - every notebook imports
from `data_preprocessing.py`, `feature_engineering.py`, `model.py`, and
`train.py` rather than duplicating logic. There is no single
`python src/train.py` entry point; run the notebooks in order instead.

---

## Key Technical Decisions

| Decision | Choice | Justification |
|---|---|---|
| Loss Function | Huber Loss | Target skewness 3.39 - MSE dominated by consumption spikes |
| Input Scaling | StandardScaler | Sensor features approximately normally distributed |
| Target Scaling | MinMaxScaler | Applied after outlier clipping for clean 0-1 output range |
| Outlier Treatment | IQR clipping | Row removal breaks temporal continuity for rolling/lag features |
| Sequence Window | Hyperparameter [24, 72, 144] | ACF confirms daily cycle; optimal length validated empirically |
| Dropout | SpatialDropout1D | Preserves temporal structure; standard dropout disrupts RNN flow |
| Optimizer | Adam + Cosine Annealing LR | Smooth decay avoids getting stuck in local minima |
| Primary DL Model | CNN-LSTM + Attention | Local pattern extraction + sequence memory + selective temporal focus |

---

## Results

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Linear Regression | 13.51 | 21.30 | 17.61 | 0.696 |
| Random Forest | 16.54 | 23.18 | 21.93 | 0.640 |
| GRU | 17.40 | 25.77 | 22.00 | 0.551 |
| CNN-LSTM | 18.35 | 25.95 | 24.49 | 0.544 |
| LSTM | 17.45 | 26.10 | 22.27 | 0.539 |
| CNN-LSTM + Attention | 18.07 | 26.17 | 23.56 | 0.537 |
| TCN | 19.55 | 27.33 | 26.19 | 0.495 |

Full table with every optimization/ablation variant:
[`reports/final_results_table.csv`](reports/final_results_table.csv).

---

## Further Optimization & Ceiling Analysis

Linear Regression outperforming every deep learning architecture is a
genuine, investigated finding, not an unaddressed weakness. Notebook 05
(sections 9-17) runs a battery of independent tests to determine whether
that gap is fixable or a real ceiling set by the data:

| Test | Result | Verdict |
|---|---|---|
| Raw-features-only ablation | GRU R² collapses to **-0.09** without the target's own lag/rolling features | Almost all signal is autoregressive |
| ARIMA(2,1,2), target history only | RMSE 21.64, R² 0.686 - ties Linear Regression | Ceiling isn't specific to the DL architectures |
| XGBoost, same 32 features | RMSE 21.32, R² 0.696 - ties Linear Regression | Nonlinear interactions add little once `Appliances_lag1` is available |
| GRU hyperparameter random search | RMSE 25.77 → 24.96 (~3%) | Capacity was a minor factor, not the main one |
| Engineered feature set v2 (interactions, trimmed lags, composites) | RMSE unchanged (25.77) | Neutral |
| Log-transformed target | Hurts Linear Regression (21.30→22.89), helps GRU slightly (25.77→25.39) | Huber loss was already the better fix for skew |
| Lagged exogenous sensors (thermal inertia) | RMSE 25.77 → 25.42 (~1%) | Small, real, not a gap-closer |
| **Stacking: Linear Regression + GRU on residuals** | **RMSE 21.06, R² 0.6999 - beats Linear Regression** | **The one approach that actually wins** |

Four independent methods (ARIMA, XGBoost, the raw-features ablation, and
the shared ceiling across all 5 DL architectures) converge on the same
conclusion: `Appliances` at 10-minute resolution is close to a pure
autoregressive process. No amount of extra feature engineering, model
capacity, or target transformation closes the gap for a standalone deep
sequence model - full details and numbers in `notebooks/05_Optimization_Evaluation.ipynb`.
The one genuine win came from **combining** Linear Regression with a GRU
to correct its residuals, not from replacing it with a bigger model.

---

## Known Environment Issue: PyTorch + XGBoost

On some machines (confirmed on macOS/Intel here), importing PyTorch and
XGBoost into the **same process** segfaults once PyTorch has done real
training work - both libraries bundle their own copy of the OpenMP
runtime (`libomp.dylib`), and the two conflict. `n_jobs=1` on
`XGBRegressor` is not a reliable fix by itself once torch's runtime is
already active. The working fix, applied in `05_Optimization_Evaluation.ipynb`:
XGBoost runs in a fully separate subprocess via `src/run_xgboost_subprocess.py`
(which never imports torch), invoked with `subprocess.run([sys.executable,
"-m", "src.run_xgboost_subprocess"], ...)` and its JSON output parsed back
into the notebook. If you extend this project and use XGBoost elsewhere,
keep it out of any process that also imports torch.

---

## AI Tools Declaration

Claude (Anthropic) was used during this assessment for:
- Architecture planning and design decisions
- Code structure and docstring guidance
- Debugging and review

All implementations, analysis, interpretations, and final decisions were made
with full understanding by the candidate. Every line of code can be explained.

---

## Dependencies

See `requirements.txt` for pinned versions. Key libraries:

- Python 3.7+
- TensorFlow >= 2.13
- Scikit-learn >= 1.3
- Pandas >= 2.0
- Statsmodels >= 0.14
- Matplotlib >= 3.7
- Seaborn >= 0.12