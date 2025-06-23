# Product Rationalization AI System - README

## Overview

This project is a comprehensive **Streamlit-based AI application** that performs product rationalization using synthetic or real e-commerce data. It covers data generation, preprocessing, pricing optimization, sales forecasting, performance evaluation, and insight generation with business recommendations.

---

## Modules Breakdown

### 1. `DataGenerator`

**Purpose:** Creates synthetic e-commerce records.

**Key Features:**

* Randomized pricing with Gaussian variation.
* Promotion influence on demand (`promo_multiplier = 1.3 if promo else 1.0`).
* Price elasticity effect simulated using:
  $\text{price\_effect} = (\frac{\text{price}}{\text{base\_price}})^{-0.5}$
* Seasonal variation using sinusoidal trend.
* Page views estimated using:
  $\text{views} \sim \text{Uniform}(2, 5) \times \text{sales} + N(0, 10)$
* Shipping cost by country with jitter.
* Profit: $\text{gross\_profit} = \text{sales} \times \text{price} \times \text{margin}$

---

### 2. `DataPreprocessor`

**Purpose:** Robust cleaning and feature engineering.

**Key Features:**

* Flexibly detects date, price, sales, views columns by heuristics.
* Fills missing values with medians or generated estimates.
* Generates derived features:

  * `revenue = sales_qty * price`
  * `month`, `quarter`, `is_weekend`
  * `price_elasticity`: $\%\Delta \text{price}$ over time by product

**Enhancement Ideas:**

* Impute missing categorical values using mode or predictive modeling.
* Create lag features or rolling aggregates.

---

### 3. `PriceOptimizer` (Standard)

**Purpose:** Estimate price elasticity from data and optimize pricing.

**Logic:**

* Calculates elasticity using correlation of % price change and % quantity change.
  $\text{Elasticity} = corr(\%\Delta \text{price}, \%\Delta \text{quantity})$
* Uses `scipy.optimize.minimize_scalar` to maximize revenue or profit.
* Demand model: $q = q_{avg} \cdot (\text{price multiplier})^{\text{elasticity}}$

**Enhancements:**

* Switch to improved `PriceOptimizer` using demand curve regression (see below).
* Use ensemble-based demand modeling instead of fixed elasticity.

---

### 4. `ForecastingEngine`

**Purpose:** Predict future demand using:

* LSTM (neural net)
* ARIMA (time series model)
* Fallback: simple average + seasonality trend

**LSTM Logic:**

* Takes 30-day sliding windows of sales.
* Predicts next value via sequential modeling.

**ARIMA Logic:**

* (2,1,2) ARIMA captures short-term autocorrelation and trends.

**Enhancements:**

* Use Prophet or XGBoost with calendar events.
* Include features like `promo_flag`, `day_of_week` in multivariate models.

---

### 5. `LLMInsightGenerator`

**Purpose:** Simulate LLM-driven business insight and recommendation generation.

**Generated Insights:**

1. **Revenue Summary**: Total and average margin.
2. **Top Performers**: Products with highest revenue.
3. **Category Leaders**: Best-performing categories.
4. **Seasonality**: Month with peak average sales.
5. **Promo Impact**: Uplift in sales when promotions are active.

**Generated Recommendations:**

* Use price optimization for elastic products.
* Forecast-driven stock planning.
* Focus on high-margin + high-conversion SKUs.
* Investigate low-conversion/high-traffic products.
* Expand into geographies with low shipping cost and high demand.

**Enhancements:**

* Use real OpenAI or LLaMA-3 model for custom insight generation.
* Add feedback loop to prioritize or score each insight.

---

### 6. `ProductRationalizationEngine`

**Purpose:** Integrates all components and drives final decisions.

**Product Scoring:**

* `revenue_score = product_revenue / max_revenue`
* `profit_score = product_profit / max_profit`
* `engagement_score = avg_views / max_views`
* Final score = $0.4 \cdot \, \text{revenue} + 0.4 \cdot \, \text{profit} + 0.2 \cdot \, \text{engagement}$

**Actions Based on Score:**

* $\text{Score} \geq 0.7 \Rightarrow \text{PROMOTE}$
* $0.4 \leq \text{Score} < 0.7 \Rightarrow \text{KEEP}$
* $\text{Score} < 0.4 \Rightarrow \text{REVIEW}$

**Enhancements:**

* Integrate classification model to learn optimal thresholds.
* Show rationale in a sentence (e.g., “High profit but low engagement”).

---


