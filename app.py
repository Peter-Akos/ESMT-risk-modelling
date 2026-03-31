import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import norm


st.set_page_config(
    page_title="Dynamic Risk Portfolio Analyzer",
    page_icon="📉",
    layout="wide",
)


TRADING_DAYS = 252
PERIOD_MAP = {
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "10y": "10y",
}
PERIOD_TRADING_DAYS_EST = {
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "10y": 2520,
}
TICKER_OPTIONS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "XOM",
    "BTC-USD",
    "ETH-USD",
    "^GSPC",
    "^IXIC",
    "^DJI",
]
DEFAULT_INPUTS = {
    "ticker_input": "AAPL",
    "period_label_input": "2y",
    "confidence_pct_input": 95,
    "horizon_days_input": 10,
    "n_sims_input": 5000,
    "portfolio_value_input": 100_000.0,
    "backtest_lookback_input": 252,
}


@st.cache_data(show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
        data = data["Close"]
    elif "Close" in data.columns:
        data = data["Close"]
    return to_1d_series(data)


def to_1d_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return pd.Series(dtype=float)
        data = data.iloc[:, 0]
    return pd.to_numeric(data, errors="coerce").dropna()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    prices = to_1d_series(prices)
    return np.log(prices / prices.shift(1)).dropna()


def aggregated_log_returns(log_returns: pd.Series, horizon_days: int) -> pd.Series:
    log_returns = to_1d_series(log_returns)
    return log_returns.rolling(window=horizon_days).sum().dropna()


def parametric_var_cvar(
    log_returns: pd.Series, confidence: float, horizon_days: int
) -> tuple[float, float]:
    log_returns = to_1d_series(log_returns)
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std(ddof=1)
    mu_h = mu_daily * horizon_days
    sigma_h = sigma_daily * np.sqrt(horizon_days)
    z_left = norm.ppf(1 - confidence)
    var = -(mu_h + sigma_h * z_left)
    cvar = -mu_h + sigma_h * (norm.pdf(z_left) / (1 - confidence))
    return max(var, 0.0), max(cvar, 0.0)


def historical_var_cvar(
    log_returns: pd.Series, confidence: float, horizon_days: int
) -> tuple[float, float, pd.Series]:
    hist_h = aggregated_log_returns(log_returns, horizon_days)
    losses = -hist_h
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean()
    return max(float(var), 0.0), max(float(cvar), 0.0), losses


def monte_carlo_gbm(
    prices: pd.Series,
    log_returns: pd.Series,
    confidence: float,
    horizon_days: int,
    n_sims: int,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)
    prices = to_1d_series(prices)
    log_returns = to_1d_series(log_returns)
    s0 = float(prices.iloc[-1])
    sigma_annual = log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    mu_annual = (log_returns.mean() * TRADING_DAYS) + (0.5 * sigma_annual**2)

    dt = 1 / TRADING_DAYS
    shocks = np.random.randn(n_sims, horizon_days)
    log_increments = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(
        dt
    ) * shocks

    cum_log_returns = np.cumsum(log_increments, axis=1)
    paths = s0 * np.exp(cum_log_returns)
    final_prices = paths[:, -1]

    sim_log_returns = np.log(final_prices / s0)
    losses = -sim_log_returns
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean()

    return {
        "s0": s0,
        "paths": paths,
        "final_prices": final_prices,
        "losses": losses,
        "var": max(float(var), 0.0),
        "cvar": max(float(cvar), 0.0),
        "mu_annual": float(mu_annual),
        "sigma_annual": float(sigma_annual),
    }


def backtest_parametric_var(
    log_returns: pd.Series, confidence: float, horizon_days: int, lookback: int
) -> pd.DataFrame:
    log_returns = to_1d_series(log_returns)
    rows = []
    z_left = norm.ppf(1 - confidence)
    max_i = len(log_returns) - horizon_days + 1

    for i in range(lookback, max_i):
        window = log_returns.iloc[i - lookback : i]
        realized_h = log_returns.iloc[i : i + horizon_days].sum()
        mu_w = window.mean() * horizon_days
        sigma_w = window.std(ddof=1) * np.sqrt(horizon_days)
        var_t = -(mu_w + sigma_w * z_left)
        loss_realized = -realized_h
        rows.append(
            {
                "Date": log_returns.index[i + horizon_days - 1],
                "VaR": max(float(var_t), 0.0),
                "RealizedLoss": max(float(loss_realized), 0.0),
            }
        )

    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["Breach"] = bt["RealizedLoss"] > bt["VaR"]
    return bt


def pct(x: float) -> str:
    return f"{x:.2%}"


def usd(amount: float) -> str:
    return f"${amount:,.2f}"


def assumptions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Method": "Parametric VaR (Variance-Covariance)",
                "Strengths": "Fast and interpretable; closed-form tail metrics.",
                "Limitations": "Assumes normal returns and can underestimate fat tails.",
            },
            {
                "Method": "Historical Simulation",
                "Strengths": "No distribution assumption; uses realized market behavior.",
                "Limitations": "Backward-looking; misses unseen regimes and new shocks.",
            },
            {
                "Method": "Monte Carlo (GBM)",
                "Strengths": "Flexible scenario generation and intuitive path visualization.",
                "Limitations": "Model risk from GBM assumptions; compute-heavy at high N.",
            },
        ]
    )


st.title("Dynamic Risk Portfolio Analyzer")
st.caption(
    "Interactive VaR/CVaR dashboard with Parametric, Historical, and Monte Carlo risk engines."
)

for key, value in DEFAULT_INPUTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("User Inputs")
    ticker = st.selectbox(
        "Ticker Selection",
        options=TICKER_OPTIONS,
        key="ticker_input",
        help="Choose a ticker to analyze.",
    )
    period_label = st.selectbox(
        "Historical Window",
        options=list(PERIOD_MAP.keys()),
        key="period_label_input",
        help="Lookback window used for parameter estimation and backtesting history.",
    )
    confidence_pct = st.slider(
        "Confidence Interval (α)",
        min_value=90,
        max_value=99,
        key="confidence_pct_input",
        help="Higher confidence implies a more conservative (larger) VaR.",
    )
    horizon_days = st.number_input(
        "Time Horizon (days)",
        min_value=1,
        max_value=60,
        key="horizon_days_input",
        help="Risk horizon in trading days (for example: 1-day vs 10-day VaR).",
    )
    n_sims = st.slider(
        "Monte Carlo Simulations (N)",
        min_value=1000,
        max_value=10000,
        step=500,
        key="n_sims_input",
        help="More paths improve stability but increase runtime.",
    )
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=1000.0,
        max_value=50_000_000.0,
        step=1000.0,
        key="portfolio_value_input",
        help="Used to convert percentage VaR/CVaR into dollar terms.",
    )
    backtest_lookback = st.slider(
        "Backtesting Lookback (days)",
        min_value=126,
        max_value=756,
        step=21,
        key="backtest_lookback_input",
        help="Rolling window length used to forecast VaR in the backtest.",
    )
    ticker_clean = ticker.strip().upper()
    required_points = max(int(backtest_lookback) + int(horizon_days) + 5, 130)
    available_est = PERIOD_TRADING_DAYS_EST[period_label]
    period_feasible = available_est >= required_points

    if not period_feasible:
        st.warning(
            f"Selected period may be too short for these settings (needs ~{required_points} days). "
            "Increase historical window or lower lookback/horizon."
        )

    run = st.button("Run Analysis", type="primary", disabled=not period_feasible)

tab_engine, tab_theory = st.tabs(["Live Engine", "Learning Lab"])

with tab_theory:
    st.subheader("Theory & Education")
    st.markdown("### Return Definition")
    st.latex(r"r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")

    st.markdown("### Historical VaR / CVaR")
    st.markdown("Define losses as positive values:")
    st.latex(r"L = -R")
    st.latex(r"\mathrm{VaR}_{\alpha} = \mathrm{Percentile}(L,\alpha)")
    st.latex(r"\mathrm{CVaR}_{\alpha} = \mathbb{E}\left[L \mid L \ge \mathrm{VaR}_{\alpha}\right]")

    st.markdown("### Parametric VaR / CVaR (Normal assumption)")
    st.latex(r"R_h \sim \mathcal{N}\!\left(\mu_h,\sigma_h^2\right),\quad \mu_h=h\mu,\ \sigma_h=\sqrt{h}\sigma")
    st.latex(r"z_{1-\alpha}=\Phi^{-1}(1-\alpha)")
    st.latex(r"\mathrm{VaR}_{\alpha}^{\mathrm{para}} = -\left(\mu_h + \sigma_h z_{1-\alpha}\right)")
    st.latex(
        r"\mathrm{CVaR}_{\alpha}^{\mathrm{para}} = -\mu_h + \sigma_h\frac{\phi(z_{1-\alpha})}{1-\alpha}"
    )

    st.markdown("### Geometric Brownian Motion")
    st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t")
    st.latex(
        r"S_{t+1}=S_t\exp\!\left[\left(\mu-\frac{1}{2}\sigma^2\right)\Delta t+\sigma\sqrt{\Delta t}\,\varepsilon_t\right],\ \varepsilon_t\sim\mathcal{N}(0,1)"
    )
    st.markdown("#### Assumptions Log")
    st.dataframe(assumptions_table(), use_container_width=True, hide_index=True)
    st.info(
        "Kurtosis tooltip: high kurtosis means fatter tails, so extreme losses are more likely than under a normal distribution."
    )

with tab_engine:
    st.subheader("Analysis & Visualization")
    if not run:
        st.write("Set inputs in the sidebar and click **Run Analysis**.")
    else:
        with st.spinner("Downloading data and running risk models..."):
            prices = load_prices(ticker_clean, PERIOD_MAP[period_label])
            if prices.empty:
                st.error(f"No price data found for `{ticker_clean}`.")
                st.stop()

            log_returns = compute_log_returns(prices)
            if len(log_returns) < max(backtest_lookback + horizon_days + 5, 130):
                st.error(
                    "Not enough historical data for the selected settings. "
                    "Try a longer historical window or lower backtesting lookback."
                )
                st.stop()

            conf = confidence_pct / 100.0

            p_var, p_cvar = parametric_var_cvar(log_returns, conf, int(horizon_days))
            h_var, h_cvar, hist_losses = historical_var_cvar(log_returns, conf, int(horizon_days))
            mc = monte_carlo_gbm(
                prices=prices,
                log_returns=log_returns,
                confidence=conf,
                horizon_days=int(horizon_days),
                n_sims=int(n_sims),
            )
            bt = backtest_parametric_var(
                log_returns=log_returns,
                confidence=conf,
                horizon_days=int(horizon_days),
                lookback=int(backtest_lookback),
            )

        st.success(
            f"Key takeaway: at {confidence_pct}% confidence over {int(horizon_days)} days for "
            f"`{ticker_clean}`, Monte Carlo VaR is {usd(mc['var'] * portfolio_value)} ({pct(mc['var'])})."
        )

        st.markdown("#### Return Distribution vs Normal")
        mu_ret = float(log_returns.mean())
        sigma_ret = float(log_returns.std(ddof=1))
        skew_ret = float(log_returns.skew())
        kurt_ret = float(log_returns.kurt())

        x_grid = np.linspace(
            float(log_returns.min()),
            float(log_returns.max()),
            400,
        )
        normal_pdf = norm.pdf(x_grid, loc=mu_ret, scale=sigma_ret)

        fig_ret = go.Figure()
        fig_ret.add_trace(
            go.Histogram(
                x=log_returns,
                nbinsx=80,
                histnorm="probability density",
                name="Empirical Returns",
                opacity=0.7,
            )
        )
        fig_ret.add_trace(
            go.Scatter(
                x=x_grid,
                y=normal_pdf,
                mode="lines",
                name="Normal Fit",
                line=dict(color="black", width=2),
            )
        )
        fig_ret.add_vline(
            x=mu_ret,
            line_width=1.5,
            line_dash="dash",
            line_color="firebrick",
            annotation_text=f"Mean {mu_ret:.3%}",
        )
        fig_ret.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Daily Log Return",
            yaxis_title="Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.99,
                    text=f"Skewness: {skew_ret:.2f}<br>Kurtosis: {kurt_ret:.2f}",
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                )
            ],
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Parametric VaR / CVaR",
            f"{pct(p_var)} / {pct(p_cvar)}",
            help="Variance-Covariance approach under normality.",
        )
        c1.caption(f"Dollar VaR: {usd(p_var * portfolio_value)}")
        c2.metric(
            "Historical VaR / CVaR",
            f"{pct(h_var)} / {pct(h_cvar)}",
            help="Empirical tail from rolling historical horizon returns.",
        )
        c2.caption(f"Dollar VaR: {usd(h_var * portfolio_value)}")
        c3.metric(
            "Monte Carlo VaR / CVaR",
            f"{pct(mc['var'])} / {pct(mc['cvar'])}",
            help="GBM simulation using annualized drift and volatility from log returns.",
        )
        c3.caption(f"Dollar VaR: {usd(mc['var'] * portfolio_value)}")
        summary_df = pd.DataFrame(
            [
                {
                    "Ticker": ticker_clean,
                    "Method": "Parametric",
                    "Confidence": f"{confidence_pct}%",
                    "HorizonDays": int(horizon_days),
                    "VaRPercent": p_var,
                    "CVaRPercent": p_cvar,
                    "VaRDollar": p_var * portfolio_value,
                    "CVaRDollar": p_cvar * portfolio_value,
                },
                {
                    "Ticker": ticker_clean,
                    "Method": "Historical",
                    "Confidence": f"{confidence_pct}%",
                    "HorizonDays": int(horizon_days),
                    "VaRPercent": h_var,
                    "CVaRPercent": h_cvar,
                    "VaRDollar": h_var * portfolio_value,
                    "CVaRDollar": h_cvar * portfolio_value,
                },
                {
                    "Ticker": ticker_clean,
                    "Method": "MonteCarloGBM",
                    "Confidence": f"{confidence_pct}%",
                    "HorizonDays": int(horizon_days),
                    "VaRPercent": mc["var"],
                    "CVaRPercent": mc["cvar"],
                    "VaRDollar": mc["var"] * portfolio_value,
                    "CVaRDollar": mc["cvar"] * portfolio_value,
                },
            ]
        )
        st.download_button(
            "Download Results (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"risk_summary_{ticker_clean}_{period_label}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("#### Return/Price Distribution")
        fig_dist = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Historical Losses ({horizon_days}-Day Log Return Loss)",
                f"Monte Carlo Final Prices (N={n_sims:,})",
            ),
        )

        fig_dist.add_trace(
            go.Histogram(x=hist_losses, nbinsx=80, name="Historical Losses", opacity=0.75),
            row=1,
            col=1,
        )
        fig_dist.add_vline(
            x=h_var,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Hist VaR {confidence_pct}%",
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Histogram(x=mc["final_prices"], nbinsx=80, name="Final Prices", opacity=0.75),
            row=1,
            col=2,
        )
        mc_price_cutoff = mc["s0"] * np.exp(-mc["var"])
        fig_dist.add_vline(
            x=mc_price_cutoff,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"MC VaR Cutoff ({mc_price_cutoff:,.2f})",
            row=1,
            col=2,
        )
        fig_dist.update_layout(
            barmode="overlay",
            height=450,
            margin=dict(l=20, r=20, t=70, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("#### Interactive Monte Carlo Paths (Spider Web)")
        paths = mc["paths"]
        n_show = min(120, paths.shape[0])
        idx = np.random.choice(paths.shape[0], size=n_show, replace=False)
        fig_paths = go.Figure()
        for j in idx:
            fig_paths.add_trace(
                go.Scatter(
                    x=np.arange(1, int(horizon_days) + 1),
                    y=paths[j],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.2,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        median_path = np.median(paths, axis=0)
        p05 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        fig_paths.add_trace(
            go.Scatter(
                x=np.arange(1, int(horizon_days) + 1),
                y=median_path,
                mode="lines",
                line=dict(width=3, color="black"),
                name="Median Path",
            )
        )
        fig_paths.add_trace(
            go.Scatter(
                x=np.arange(1, int(horizon_days) + 1),
                y=p05,
                mode="lines",
                line=dict(width=2, color="firebrick", dash="dot"),
                name="5th Percentile",
            )
        )
        fig_paths.add_trace(
            go.Scatter(
                x=np.arange(1, int(horizon_days) + 1),
                y=p95,
                mode="lines",
                line=dict(width=2, color="seagreen", dash="dot"),
                name="95th Percentile",
            )
        )
        fig_paths.update_layout(
            xaxis_title="Day",
            yaxis_title="Simulated Price",
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        st.markdown("#### Backtesting (Parametric VaR Breaches)")
        if bt.empty:
            st.warning("Backtesting output is empty for the selected configuration.")
        else:
            breach_rate = float(bt["Breach"].mean())
            expected_rate = 1 - conf
            diff = breach_rate - expected_rate
            b1, b2, b3 = st.columns(3)
            b1.metric("Observed Breach Rate", pct(breach_rate))
            b2.metric("Theoretical Breach Rate", pct(expected_rate))
            b3.metric("Difference", pct(diff))

            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=bt["Date"],
                    y=bt["RealizedLoss"],
                    mode="lines",
                    name="Realized Loss",
                    line=dict(color="steelblue", width=1.5),
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=bt["Date"],
                    y=bt["VaR"],
                    mode="lines",
                    name="VaR Threshold",
                    line=dict(color="firebrick", width=2, dash="dash"),
                )
            )
            breaches = bt[bt["Breach"]]
            fig_bt.add_trace(
                go.Scatter(
                    x=breaches["Date"],
                    y=breaches["RealizedLoss"],
                    mode="markers",
                    name="Breach",
                    marker=dict(color="red", size=6),
                )
            )
            fig_bt.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Date",
                yaxis_title="Loss (log-return space)",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        stats_c1, stats_c2, stats_c3 = st.columns(3)
        stats_c1.metric("Latest Price", f"${prices.iloc[-1]:,.2f}")
        stats_c2.metric(
            "Annualized Volatility",
            pct(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            help="Estimated from selected lookback window.",
        )
        stats_c3.metric(
            "Kurtosis",
            f"{log_returns.kurt():.2f}",
            help="Higher values indicate fatter tails and more extreme outcomes.",
        )
