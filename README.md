# Dynamic Risk Portfolio Analyzer

Interactive Streamlit dashboard for market risk analysis using Yahoo Finance data.  
The app compares **Parametric VaR**, **Historical VaR**, and **Monte Carlo (GBM) VaR**, and also reports **CVaR (Expected Shortfall)** and backtesting breaches.

Developed for the **Risk Modelling** course.

## Author

Ákos Péter

## Features

- Live risk engine in the first tab (`Live Engine`)
- Learning/theory section in the second tab (`Learning Lab`)
- Ticker selection from a predefined list
- User controls for confidence level, horizon, lookback, simulations, and portfolio value
- Return distribution plot vs fitted normal distribution
- Skewness and kurtosis display
- Monte Carlo path plot ("spider web") and final price distribution
- Backtesting module with breach-rate comparison
- CSV export of key VaR/CVaR outputs

## Project Structure

- `app.py` - Streamlit dashboard
- `requirements.txt` - Python dependencies
- `requirements.md` - project requirements/spec
- `05_02_quant_risk_management.ipynb` - notebook used as formula reference

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Core Methodology

- **Log returns**:
  - \( r_t = \ln(P_t / P_{t-1}) \)
- **Historical VaR/CVaR**:
  - losses defined as \( L = -R \)
  - VaR is empirical percentile of losses
  - CVaR is average loss in the tail beyond VaR
- **Parametric VaR/CVaR** (normal assumption):
  - horizon scaling with \( \mu_h = h\mu \), \( \sigma_h = \sqrt{h}\sigma \)
- **Monte Carlo GBM**:
  - \( S_{t+1} = S_t \exp((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t) \)

## Notes

- Data source is Yahoo Finance via `yfinance`.
- Very short histories or incompatible horizon/lookback settings may be rejected by validation.
- Results are educational and should not be treated as investment advice.
