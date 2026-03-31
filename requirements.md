# Project Requirements Document: Dynamic Risk Portfolio Analyzer

## 1. Project Overview
This project is an interactive web-based dashboard designed to quantify and visualize financial market risk. Developed for the **Risk Modelling** course, it utilizes Python, Streamlit, and real-time market data to calculate Value at Risk (VaR) and Expected Shortfall (CVaR) using multiple quantitative methodologies.

## 2. Technical Stack
* **Language:** Python 3.x
* **Web Framework:** Streamlit
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Data Manipulation:** `pandas`, `numpy`
* **Statistical Analysis:** `scipy.stats`
* **Visualization:** `plotly` (interactive charts), `matplotlib`/`seaborn` (static distributions)

---

## 3. Functional Requirements

### 3.1 Data Acquisition Engine
* **Ticker Input:** Search box for any valid Yahoo Finance ticker (e.g., AAPL, BTC-USD, ^GSPC).
* **Historical Window:** User-selectable lookback period (e.g., 1yr, 2yr, 5yr) for calculating historical volatility and drift.
* **Return Calculation:** Automatic conversion of raw price data into log returns for statistical consistency.

### 3.2 Core Calculation Engine (VaR Methodologies)
The system must calculate and compare the following:
* **Parametric VaR:** Calculation using the Variance-Covariance method (Normal Distribution assumption).
* **Historical Simulation:** Non-parametric calculation based on actual historical return percentiles.
* **Monte Carlo Simulation:** * Simulate $N$ price paths using Geometric Brownian Motion (GBM).
    * Parameters: $\mu$ (drift), $\sigma$ (volatility), $T$ (time horizon), and $S_0$ (initial price).

### 3.3 Advanced "Pro" Features
* **A. Backtesting Module:**
    * Calculate the VaR threshold for a historical window.
    * Identify "Breaches" (actual losses exceeding predicted VaR).
    * Report the **Breach Rate** and compare it against the theoretical confidence level.
* **B. Expected Shortfall (CVaR):**
    * Calculate the average loss in the worst $\alpha\%$ of cases.
    * Present CVaR as a more conservative risk metric compared to VaR.
* **D. Interactive Monte Carlo Visualization:**
    * Plotly-based "Spider Web" chart showing multiple simulated price trajectories.
    * Histogram of "Final Prices" at the end of the time horizon $T$, highlighting the VaR cutoff line.

---

## 4. UI/UX Structure (The Dashboard)

### Section 1: Theory & Education (The "Learning Lab")
This separate section acts as a digital textbook for the course material.
* **Latex Explanations:** Formal definitions of VaR, CVaR, and the GBM equation.
* **Assumptions Log:** A table highlighting the pros/cons of each model (e.g., "Parametric ignores Fat Tails," "Historical ignores regime shifts").

### Section 2: User Inputs (Sidebar)
* **Ticker Selection.**
* **Confidence Interval ($\alpha$):** Slider (90%, 95%, 99%).
* **Time Horizon ($T$):** Input for number of days (e.g., 1-day vs 10-day VaR).
* **Simulations ($N$):** Slider for Monte Carlo iterations (1,000 to 10,000).

### Section 3: Analysis & Visualization (Main Panel)
* **Risk Metrics Cards:** Large metrics showing VaR in both percentage and dollar terms.
* **Distribution Plot:** A histogram of returns/prices with shaded "Danger Zones."
* **Path Plot:** The Monte Carlo trajectory chart.

---

## 5. Non-Functional Requirements
* **Accuracy:** Statistical calculations must align with standard Risk Management formulas.
* **Performance:** Monte Carlo simulations should optimize `numpy` vectorization to ensure the UI remains responsive for up to 10,000 paths.
* **Clarity:** Use "Info" tooltips to explain complex terms (e.g., explaining what "Kurtosis" means for the user's chosen stock).

## 6. Success Criteria
* Successful rendering of the Monte Carlo distribution.
* The Backtesting module correctly identifies the number of historical breaches.
* The UI provides a seamless transition between the educational "Theory" section and the "Live Engine."