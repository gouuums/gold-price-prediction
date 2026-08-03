# Modelling and forecasting gold prices: Econometric, Machine Learning, and portfolio approaches

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Models-orange.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-ff6f00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Authors:** Thomas Goumont, Tom O'Connor, Antonin Meudic, Adrien Gibert  
> **Institution:** ENSAI (École Nationale de la Statistique et de l'Analyse de l'Information)  
> **Specialisation:** Risk Management / Advanced Statistical Engineering  
> **Supervisor:** Youssef Esstafa  
> **Academic Year:** 2025–2026
---

## 📋 Project Overview

This repository contains the final report, code, and presentation materials for the advanced time series project at ENSAI. The study investigates the stochastic dynamics, statistical diagnostics, predictive modelling, and portfolio applications of gold prices using a two-decade dataset (**2004–2025**) centered around the **SPDR Gold Shares ETF (GLD)**.

The project addresses the notorious low signal-to-noise ratio in financial returns by combining classical financial econometrics (GARCH volatility modeling) with state-of-the-art machine learning (XGBoost, ARIMA-XGBoost hybrids) and deep learning architectures (LSTM networks with volatility targeting).

---

## 🔑 Key Methodological Steps & Findings

1. **Data Preprocessing & Exploratory Analysis**
   - **Dataset:** ~5,300 daily closing price observations of GLD (serving as a spot gold proxy).
   - **Stylized Facts:** Confirmed that gold log-returns approximate a stationary white noise process in the conditional mean ($ARMA(0,0)$ selected via AIC), but exhibit pronounced volatility clustering, heavy tails (excess kurtosis $\approx 8.865$), and negative skewness.
   - **Seasonality & Regimes:** Demonstrated through spectral/seasonal plots and ANOVA tests ($p = 0.59$) that gold displays no exploitable calendar seasonality. Dynamics are instead driven by macroeconomic regime shifts (e.g., post-2008 bull market, 2013 tapering crash, and the 2024–2025 inflation/geopolitical surge).

2. **Econometric Risk Modeling (GARCH Family)**
   - Evaluated GARCH-family specifications for variance persistence.
   - Selected the symmetric **GARCH(1,1) model with Student-t innovations** as optimal due to its superior out-of-sample variance forecast performance (lowest QLIKE loss), capturing fat tails and high persistence ($\alpha + \beta \approx 0.995$).

3. **Deep Learning: LSTM-GARCH Framework**
   - Designed a stacked LSTM network utilizing macro-financial innovations (10-year real yields, breakeven inflation, VIX changes, and USD Index log-differences) over a 250-day rolling look-back window.
   - Integrated LSTM return directional forecasts with GARCH conditional volatility estimates in a **risk-targeted long/short trading strategy**.
   - **Performance:** Delivered a 5-year average annual return (CAGR) of **11.25%** with an annualized volatility of **10.09%**, a **Sharpe ratio of 1.09**, and a maximum drawdown capped at **-7.22%**.

4. **Gradient Boosting & Hybrid ARIMA-XGBoost**
   - **Standalone XGBoost:** Evaluated on a 10-day cumulative return horizon to increase the signal-to-noise ratio, achieving a high directional accuracy of **76.11%**.
   - **Hybrid ARIMA-XGBoost:** Deployed an ARIMA(0,0,0) baseline to capture the global mean, paired with an XGBoost regressor trained on weekly residuals using technical indicators (RSI, Bollinger Bands, MACD) and macro features. This hybrid approach significantly outperformed standalone linear models, yielding out-of-sample directional accuracy up to **70.00%** and reducing return RMSE.

5. **Multi-Asset Portfolio Allocation**
   - Constructed a diversified tactical asset allocation portfolio (S&P 500, EUR/USD, Gold, US Treasury 20Y Bonds, CAC 40, Crude Oil) using a Markowitz Mean-Variance framework with turnover buffers and transaction cost management.

---

## 🗂️ Repository Structure

- `ST_report.pdf`: Full academic research report containing detailed mathematical formulations, proofs, tables, and econometric proofs.
- `TS_presentation.pdf`: Slide deck used for the final project defense.
- Source code notebooks: Implementation files for data pipelines, XGBoost tuning, LSTM training, and portfolio backtesting.

---

## 📦 Installation & Dependencies

To replicate the models and run the analysis, install the required packages:

```bash
pip install numpy pandas scikit-learn xgboost tensorflow optuna yfinance pandas_datareader seaborn matplotlib tqdm shap
```

## 📊 Summary Performance Metrics (Portfolio Backtest 2021–2025)

| Metric | LSTM-GARCH Strategy | Hybrid Portfolio Strategy |
|---|---|---|
| CAGR | 11.25% | 13.55% |
| Annualized Volatility | 10.09% | 11.97% |
| Sharpe Ratio | 1.09 | 0.97 |
| Max Drawdown | -7.22% | -9.50% |

## 📄 References

Key literature referenced in the study includes:

- Cont, R. (2001) - Empirical properties of asset returns: stylized facts and statistical issues.
- Jabeur, S.B. et al. (2021) - Forecasting gold price with the XGBoost algorithm and SHAP interaction values.
- Amini, A. & Kalantari, R. (2024) - Gold Price Prediction by a CNN-Bi-LSTM Model.
