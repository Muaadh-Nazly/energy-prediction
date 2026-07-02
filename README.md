# Appliance Energy Prediction
### Multivariate Time-Series Deep Learning - Take-Home Assessment

**Author:** [Your Name]
**Date:** [Date]
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

---

## Repository Structure
```
energy-prediction/
├── data/
│   ├── raw/                        - Original dataset (energy_data_set.csv)
│   └── processed/                  - Cleaned and feature-engineered outputs
├── notebooks/
│   ├── 01_EDA.ipynb                - Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_Models.ipynb
│   ├── 04_Deep_Learning_Models.ipynb
│   └── 05_Optimization_Evaluation.ipynb
├── src/
│   ├── data_preprocessing.py       - Loading, cleaning, scaling functions
│   ├── feature_engineering.py      - Feature creation and selection functions
│   ├── model.py                    - All model architecture definitions
│   └── train.py                    - Training loop and evaluation pipeline
├── models/                         - Saved trained model files (.h5 / .pt)
├── reports/
│   ├── figures/                    - All generated plots
│   └── report.pdf                  - Final assessment report
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Environment Setup

### Option 1 - Local with venv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-prediction.git
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
04_Deep_Learning_Models.ipynb - LSTM, GRU, CNN-LSTM, TCN, CNN-LSTM+Attention
05_Optimization_Evaluation.ipynb - Hyperparameter tuning, final evaluation, residual diagnostics

Alternatively, run the full training pipeline from terminal:

```bash
python src/train.py
```

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

*(Updated after training is complete)*

| Model | MAE (Wh) | RMSE (Wh) | MAPE (%) | R² |
|---|---|---|---|---|
| Linear Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| LSTM | - | - | - | - |
| GRU | - | - | - | - |
| CNN-LSTM | - | - | - | - |
| TCN | - | - | - | - |
| CNN-LSTM + Attention | - | - | - | - |

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