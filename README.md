# Product Rationalization 


## MConcept: 

### 1. `DataGenerator`

 Creates synthetic data (just for testing).


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


* detects date, price, sales, views columns .
* Fills missing values with medians or generated estimates.
* Generates derived features:

  * `revenue = sales_qty * price`
  * `month`, `quarter`, `is_weekend`
  * `price_elasticity`: $\%\Delta \text{price}$ over time by product

**possible improvement:**

* Impute missing categorical values using mode or predictive modeling.
* Create lag features or rolling aggregates.

---

### 3. `PriceOptimizer` 
 Estimate price elasticity from data and optimize pricing.

**Logic:**

* Calculates elasticity using correlation of % price change and % quantity change.
  $\text{Elasticity} = corr(\%\Delta \text{price}, \%\Delta \text{quantity})$
* Uses `scipy.optimize.minimize_scalar` to maximize revenue or profit.
* Demand model: $q = q_{avg} \cdot (\text{price multiplier})^{\text{elasticity}}$

**possible improvement:**

* Switch to improved `PriceOptimizer` using demand curve regression (see below).

class PriceOptimizer:
    
    def __init__(self, data):
        self.data = data
        
    def calculate_demand_curve(self, product_id):
        """Calculate demand curve parameters using price binning and linear regression"""
        product_data = self.data[self.data['product_id'] == product_id].copy()
        
        if len(product_data) < 10:
            return {'intercept': 100, 'slope': -2.0, 'r_squared': 0.3}
        
        # Remove promo data for better price elasticity calculation
        normal_data = product_data[product_data['promo_flag'] == 0].copy()
        
        if len(normal_data) < 5:
            normal_data = product_data.copy()
        
        # Create price bins 
        normal_data['price_decile'] = pd.qcut(normal_data['price'], 
                                             q=min(5, len(normal_data)//3), 
                                             duplicates='drop')
        
        demand_by_price = normal_data.groupby('price_decile').agg({
            'price': 'mean',
            'sales_qty': 'mean'
        }).dropna()
        
        if len(demand_by_price) < 3:
            return {'intercept': normal_data['sales_qty'].mean() or 20, 
                   'slope': -1.5, 'r_squared': 0.4}
        
        prices = demand_by_price['price'].values
        quantities = demand_by_price['sales_qty'].values
        
        # Linear regression for demand curve
        n = len(prices)
        if n < 2:
            return {'intercept': quantities.mean() or 20, 
                   'slope': -1.5, 'r_squared': 0.4}
            
        sum_xy = np.sum(prices * quantities)
        sum_x = np.sum(prices)
        sum_y = np.sum(quantities)
        sum_x2 = np.sum(prices**2)
        
        if n * sum_x2 - sum_x**2 == 0:
            return {'intercept': quantities.mean() or 20, 
                   'slope': -1.5, 'r_squared': 0.4}
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_pred = intercept + slope * prices
        ss_res = np.sum((quantities - y_pred)**2)
        ss_tot = np.sum((quantities - np.mean(quantities))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Ensure reasonable values
        intercept = max(intercept, quantities.mean() * 0.5)
        slope = min(slope, -0.1)  # Ensure negative slope but not too steep
        r_squared = max(min(r_squared, 1.0), 0.0)
        
        return {
            'intercept': intercept,
            'slope': slope,
            'r_squared': r_squared
        }
    
    def optimize_price(self, product_id, current_price, target='revenue'):
        """Find optimal price using demand curve and margin analysis"""
        product_data = self.data[self.data['product_id'] == product_id]
        
        if len(product_data) == 0:
            return current_price, "Insufficient data"
        
        demand_params = self.calculate_demand_curve(product_id)
        
        # Calculate margin 
        total_revenue = (product_data['sales_qty'] * product_data['price']).sum()
        total_profit = product_data['Gross Profit'].sum()
        avg_margin = total_profit / total_revenue if total_revenue > 0 else 0.15
        avg_margin = max(avg_margin, 0.05)  # Ensure minimum 5% margin
        
        cost_per_unit = current_price * (1 - avg_margin)
        
        def demand_function(price):
            return max(0, demand_params['intercept'] + demand_params['slope'] * price)
        
        def objective(price_multiplier):
            price = current_price * price_multiplier
            quantity = demand_function(price)
            
            if target == 'revenue':
                return -(price * quantity)  # Negative for minimization
            else:  # profit
                profit = (price - cost_per_unit) * quantity
                return -profit
        
        # Conservative price bounds (30% decrease to 40% increase)
        price_bounds = (0.7, 1.4)
        
        try:
            result = minimize_scalar(objective, bounds=price_bounds, method='bounded')
            optimal_multiplier = result.x
            optimal_price = current_price * optimal_multiplier
        except:
            optimal_price = current_price
        
        # Calculate demand elasticity
        if current_price > 0:
            current_qty = demand_function(current_price)
            elasticity = (demand_params['slope'] * current_price) / current_qty if current_qty > 0 else -1.5
        else:
            elasticity = -1.5
        
        return (round(optimal_price, 2), 
                f"Elasticity: {elasticity:.2f}, R²: {demand_params['r_squared']:.2f}")
                
* Use ensemble-based demand modeling instead of fixed elasticity.

---

### 4. `ForecastingEngine`

**Purpose:** Predict future demand using:

* LSTM 
* ARIMA (time series model)
* pretty basic: simple average + seasonality trend

**LSTM Logic:**

* Takes 30-day sliding windows of sales.
* Predicts next value via sequential modeling.

**ARIMA Logic:**

* (2,1,2) ARIMA captures short-term autocorrelation and trends.

**Enhancements:**

* Use Prophet or XGBoost with calendar events.
* Include features `promo_flag`, `day_of_week` in multivariate models.
* Use a loop to try different ARIMA parameter combinations and pick the best one using evaluation techniques (I will try and suggest which one to use)

---

### 5. `LLMInsightGenerator`

These are just mock insights, but and LLM can be integrated over the models to generate actual insights


* Use real OpenAI or LLaMA-3 model for custom insight generation.
* Add feedback loop to prioritize or score each insight.

---

### 6. `ProductRationalizationEngine`


**Product Scoring:**

* `revenue_score = product_revenue / max_revenue`
* `profit_score = product_profit / max_profit`
* `engagement_score = avg_views / max_views`
* Final score = $0.4 \cdot \, \text{revenue} + 0.4 \cdot \, \text{profit} + 0.2 \cdot \, \text{engagement}$

**Actions Based on Score:**

* $\text{Score} \geq 0.7 \Rightarrow \text{PROMOTE}$
* $0.4 \leq \text{Score} < 0.7 \Rightarrow \text{KEEP}$
* $\text{Score} < 0.4 \Rightarrow \text{REVIEW}$

**possible improvements:**

* Integrate classification model to learn optimal thresholds.
* Show rationale in a sentence (e.g., “High profit but low engagement”).

---


