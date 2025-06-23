

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize_scalar
import openai
import os


st.set_page_config(page_title="Product Rationalization AI", layout="wide", page_icon="📊")

class DataGenerator:
    """Generate synthetic data for testing"""
    
    @staticmethod
    def generate_synthetic_data(num_records=1000, start_date='2024-01-01'):
        np.random.seed(42)
        
        # Product catalog
        products = {
            'PRD123': {'category': 'Sneakers', 'base_price': 19.99, 'margin_base': 0.12},
            'PRD256': {'category': 'Sneakers', 'base_price': 15.23, 'margin_base': 0.15},
            'PRD5687': {'category': 'Gift Cards', 'base_price': 5.00, 'margin_base': 0.095},
            'PRD756': {'category': 'Sneakers', 'base_price': 5.00, 'margin_base': 0.08},
            'PRD891': {'category': 'Electronics', 'base_price': 299.99, 'margin_base': 0.25},
            'PRD442': {'category': 'Clothing', 'base_price': 45.50, 'margin_base': 0.18},
            'PRD335': {'category': 'Books', 'base_price': 12.99, 'margin_base': 0.22},
            'PRD667': {'category': 'Home', 'base_price': 89.99, 'margin_base': 0.20}
        }
        
        countries = ['Switzerland', 'India', 'UAE', 'France', 'Portugal', 'USA', 'Germany', 'UK']
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        data = []
        start = pd.to_datetime(start_date)
        
        for i in range(num_records):
            date = start + timedelta(days=np.random.randint(0, 365))
            product_id = np.random.choice(list(products.keys()))
            product_info = products[product_id]
            
            # Price with some variation
            base_price = product_info['base_price']
            price_variation = np.random.normal(1, 0.1)
            price = max(base_price * price_variation, base_price * 0.5)
            
            # Promo flag affects sales
            promo_flag = np.random.choice([0, 1], p=[0.7, 0.3])
            promo_multiplier = 1.3 if promo_flag else 1.0
            
            # Sales quantity based on price elasticity and seasonality
            base_demand = np.random.poisson(20)
            price_elasticity = -0.5
            price_effect = (price / base_price) ** price_elasticity
            seasonal_effect = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            sales_qty = max(1, int(base_demand * price_effect * promo_multiplier * seasonal_effect))
            
            # Page views correlated with sales
            page_views = max(10, int(sales_qty * np.random.uniform(2, 5) + np.random.normal(0, 10)))
            
            # Shipping cost
            country = np.random.choice(countries)
            shipping_base = {'Switzerland': 3, 'India': 2, 'UAE': 2.5, 'France': 1.5, 
                           'Portugal': 2, 'USA': 4, 'Germany': 2, 'UK': 1.8}
            shipping_cost = shipping_base.get(country, 2) + np.random.uniform(-0.5, 0.5)
            
            # Margin and profit
            margin_pct = product_info['margin_base'] + np.random.uniform(-0.05, 0.05)
            gross_profit = sales_qty * price * margin_pct
            
            # 7-day average 
            avg_7d = sales_qty * np.random.uniform(0.8, 1.2)
            
            data.append({
                'Date': date.strftime('%m/%d/%Y'),
                'product_id': product_id,
                'sales_qty': sales_qty,
                'price': round(price, 2),
                'promo_flag': promo_flag,
                '7d_avg_sales': round(avg_7d, 1),
                'day_of_week': date.strftime('%A'),
                'category': product_info['category'],
                'Page Views': page_views,
                'shipping cost': round(shipping_cost, 2),
                'Shipping Country': country,
                'Margin %': f"{margin_pct:.1%}",
                'Gross Profit': round(gross_profit, 2)
            })
        
        return pd.DataFrame(data)

class DataPreprocessor:
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def preprocess_data(self, df):
        try:
            
            processed_df = df.copy()
            
            # Handle date column (flexible column names)
            date_cols = [col for col in processed_df.columns if 'date' in col.lower()]
            if date_cols:
                processed_df['Date'] = pd.to_datetime(processed_df[date_cols[0]], errors='coerce')
            else:
                # Generate dates if missing
                processed_df['Date'] = pd.date_range(start='2024-01-01', periods=len(processed_df))
            
           
            numeric_mappings = {
                'sales_qty': ['sales', 'quantity', 'qty', 'sales_qty', 'units_sold'],
                'price': ['price', 'unit_price', 'cost'],
                'Page Views': ['page_views', 'pageviews', 'views', 'page views'],
                'shipping cost': ['shipping', 'shipping_cost', 'delivery_cost'],
                'Gross Profit': ['profit', 'gross_profit', 'revenue']
            }
            
            for standard_name, possible_names in numeric_mappings.items():
                for col in processed_df.columns:
                    if col.lower().replace('_', '').replace(' ', '') in [name.lower().replace('_', '').replace(' ', '') for name in possible_names]:
                        processed_df[standard_name] = pd.to_numeric(processed_df[col], errors='coerce')
                        break
                
                # Fill missing values with median
                if standard_name in processed_df.columns:
                    processed_df[standard_name].fillna(processed_df[standard_name].median(), inplace=True)
                else:
                    # Generate synthetic data if column missing
                    processed_df[standard_name] = np.random.poisson(10, len(processed_df))
            
            # Handle categorical columns
            categorical_mappings = {
                'category': ['category', 'product_category', 'type'],
                'Shipping Country': ['country', 'shipping_country', 'destination'],
                'day_of_week': ['day', 'weekday', 'day_of_week']
            }
            
            for standard_name, possible_names in categorical_mappings.items():
                for col in processed_df.columns:
                    if col.lower().replace('_', ' ') in [name.lower().replace('_', ' ') for name in possible_names]:
                        processed_df[standard_name] = processed_df[col].astype(str)
                        break
                
                # Fill missing categories
                if standard_name not in processed_df.columns or processed_df[standard_name].isna().any():
                    if standard_name == 'category':
                        processed_df[standard_name] = processed_df.get(standard_name, 'Unknown').fillna('Unknown')
                    elif standard_name == 'Shipping Country':
                        processed_df[standard_name] = processed_df.get(standard_name, 'USA').fillna('USA')
                    elif standard_name == 'day_of_week':
                        processed_df[standard_name] = processed_df['Date'].dt.day_name()
            
            # Handle promo flag
            if 'promo_flag' not in processed_df.columns:
                processed_df['promo_flag'] = np.random.choice([0, 1], size=len(processed_df), p=[0.7, 0.3])
            
            
            processed_df['revenue'] = processed_df['sales_qty'] * processed_df['price']
            processed_df['month'] = processed_df['Date'].dt.month
            processed_df['quarter'] = processed_df['Date'].dt.quarter
            processed_df['is_weekend'] = processed_df['Date'].dt.weekday.isin([5, 6]).astype(int)
            
            # Price elasticity feature
            processed_df['price_elasticity'] = processed_df.groupby('product_id')['price'].transform(
                lambda x: x.pct_change().fillna(0)
            )
            
            return processed_df
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return df

class PriceOptimizer:
    
    def __init__(self, data):
        self.data = data
        self.elasticity_models = {}
    
    def calculate_price_elasticity(self, product_id):
        
        product_data = self.data[self.data['product_id'] == product_id].copy()
        
        if len(product_data) < 5:
            return -1.0  # Default elasticity
        
        product_data = product_data.sort_values('Date')
        product_data['price_change'] = product_data['price'].pct_change()
        product_data['qty_change'] = product_data['sales_qty'].pct_change()
        
        # Remove outliers and NaN
        valid_data = product_data.dropna(subset=['price_change', 'qty_change'])
        valid_data = valid_data[
            (abs(valid_data['price_change']) < 0.5) & 
            (abs(valid_data['qty_change']) < 2.0)
        ]
        
        if len(valid_data) < 3:
            return -1.0
        
        # Calculate elasticity as correlation coefficient
        correlation = np.corrcoef(valid_data['price_change'], valid_data['qty_change'])[0, 1]
        return correlation if not np.isnan(correlation) else -1.0
    
    def optimize_price(self, product_id, current_price, target='revenue'):
        
        elasticity = self.calculate_price_elasticity(product_id)
        product_data = self.data[self.data['product_id'] == product_id]
        
        if len(product_data) == 0:
            return current_price, "Insufficient data"
        
        avg_qty = product_data['sales_qty'].mean()
        avg_margin = 0.15  # Default 
        
        def objective(price_multiplier):
            new_price = current_price * price_multiplier
            # Demand response to price change
            qty_response = avg_qty * (price_multiplier ** elasticity)
            
            if target == 'revenue':
                return -(new_price * qty_response)  # Negative for minimization
            else:  # profit
                profit = (new_price * avg_margin) * qty_response
                return -profit
        
        # Optimize within reasonable bounds
        result = minimize_scalar(objective, bounds=(0.5, 2.0), method='bounded')
        optimal_multiplier = result.x
        optimal_price = current_price * optimal_multiplier
        
        return round(optimal_price, 2), f"Elasticity: {elasticity:.2f}"

class ForecastingEngine:
    """Time series forecasting using LSTM and ARIMA"""
    
    def __init__(self):
        self.lstm_model = None
        self.scaler = StandardScaler()
    
    def prepare_lstm_data(self, data, sequence_length=30):
        """Prepare data for LSTM training"""
        # Aggregate daily sales
        daily_sales = data.groupby('Date')['sales_qty'].sum().reset_index()
        daily_sales = daily_sales.sort_values('Date')
        
        # Scale the data
        sales_scaled = self.scaler.fit_transform(daily_sales[['sales_qty']])
        
        X, y = [], []
        for i in range(sequence_length, len(sales_scaled)):
            X.append(sales_scaled[i-sequence_length:i, 0])
            y.append(sales_scaled[i, 0])
        
        return np.array(X), np.array(y), daily_sales
    
    def build_lstm_model(self, sequence_length=30):
        """Build LSTM model for forecasting"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def forecast_lstm(self, data, days_ahead=30):
        """Generate LSTM forecast"""
        try:
            X, y, daily_sales = self.prepare_lstm_data(data)
            
            if len(X) < 50:  # Not enough data for LSTM
                return self.simple_forecast(data, days_ahead)
            
            # Train model
            self.lstm_model = self.build_lstm_model()
            self.lstm_model.fit(X, y, epochs=50, batch_size=16, verbose=0)
            
            # Generate forecast
            last_sequence = X[-1].reshape(1, -1, 1)
            forecast = []
            
            for _ in range(days_ahead):
                pred = self.lstm_model.predict(last_sequence, verbose=0)[0, 0]
                forecast.append(pred)
                
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred
            
            # Inverse transform
            forecast_actual = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            
            return forecast_actual.flatten()
            
        except Exception as e:
            st.warning(f"LSTM forecast failed: {str(e)}. Using simple forecast.")
            return self.simple_forecast(data, days_ahead)
    
    def forecast_arima(self, data, days_ahead=30):
        """Generate ARIMA forecast"""
        try:
            daily_sales = data.groupby('Date')['sales_qty'].sum().reset_index()
            daily_sales = daily_sales.sort_values('Date')
            
            if len(daily_sales) < 30:
                return self.simple_forecast(data, days_ahead)
            
            # Fit ARIMA model
            model = ARIMA(daily_sales['sales_qty'], order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=days_ahead)
            return np.maximum(forecast, 0)  # Ensure non-negative
            
        except Exception as e:
            st.warning(f"ARIMA forecast failed: {str(e)}. Using simple forecast.")
            return self.simple_forecast(data, days_ahead)
    
    def simple_forecast(self, data, days_ahead=30):
        """Simple moving average forecast as fallback"""
        daily_sales = data.groupby('Date')['sales_qty'].sum().reset_index()
        avg_sales = daily_sales['sales_qty'].tail(30).mean()
        
        # Add some seasonality
        trend = np.linspace(0.95, 1.05, days_ahead)
        forecast = avg_sales * trend
        
        return forecast

class LLMInsightGenerator:
    """Generate business insights using LLM (mock implementation)"""
    
    def __init__(self):
        # In production, you would initialize OpenAI client here
        pass
    
    def generate_insights(self, data, analysis_results):
        """Generate business insights (mock implementation)"""
        # These are Mock insights - in production, this would use some LLLM API
        insights = []
        
        # Revenue insights
        total_revenue = data['revenue'].sum()
        avg_margin = data['Gross Profit'].sum() / total_revenue if total_revenue > 0 else 0
        
        insights.append(f"**Revenue Analysis**: Total revenue of ${total_revenue:,.2f} with average margin of {avg_margin:.1%}")
        
        # Product performance
        top_products = data.groupby('product_id')['revenue'].sum().sort_values(ascending=False).head(3)
        insights.append(f"**Top Performers**: {', '.join(top_products.index)} driving {top_products.sum()/total_revenue:.1%} of revenue")
        
        # Category insights
        category_performance = data.groupby('category')['revenue'].sum().sort_values(ascending=False)
        insights.append(f"**Category Leaders**: {category_performance.index[0]} category leads with ${category_performance.iloc[0]:,.0f}")
        
        # Seasonal patterns
        monthly_sales = data.groupby(data['Date'].dt.month)['sales_qty'].mean()
        peak_month = monthly_sales.idxmax()
        insights.append(f"**Seasonality**: Peak sales in month {peak_month} with {monthly_sales.max():.0f} avg units")
        
        # Promotional effectiveness
        promo_lift = data[data['promo_flag'] == 1]['sales_qty'].mean() / data[data['promo_flag'] == 0]['sales_qty'].mean()
        insights.append(f"**Promo Impact**: Promotions increase sales by {(promo_lift-1)*100:.1f}%")
        
        return insights
    
    def generate_recommendations(self, analysis_results):
        """Generate actionable recommendations"""
        recommendations = []
        
        if 'price_optimization' in analysis_results:
            recommendations.append("**Price Optimization**: Consider implementing dynamic pricing based on demand elasticity")
        
        if 'forecast' in analysis_results:
            recommendations.append("**Inventory Planning**: Use demand forecast to optimize stock levels and reduce waste")
        
        recommendations.append("**Marketing Focus**: Concentrate promotional efforts on high-margin, elastic products")
        recommendations.append("**Geographic Expansion**: Analyze shipping cost vs. demand patterns for market expansion")
        recommendations.append("**Digital Optimization**: Improve page experience for products with high views but low conversion")
        
        return recommendations

class ProductRationalizationEngine:
    
    def __init__(self):
        self.data_processor = DataPreprocessor()
        self.price_optimizer = None
        self.forecasting_engine = ForecastingEngine()
        self.insight_generator = LLMInsightGenerator()
    
    def analyze_products(self, data):
        results = {}
        
        # Process data
        processed_data = self.data_processor.preprocess_data(data)
        self.price_optimizer = PriceOptimizer(processed_data)
        
        # Product performance metrics
        product_metrics = processed_data.groupby('product_id').agg({
            'sales_qty': ['sum', 'mean', 'std'],
            'revenue': ['sum', 'mean'],
            'Gross Profit': 'sum',
            'Page Views': 'mean',
            'price': 'mean'
        }).round(2)
        
        product_metrics.columns = ['Total Sales', 'Avg Sales', 'Sales Volatility', 
                                 'Total Revenue', 'Avg Revenue', 'Total Profit', 
                                 'Avg Page Views', 'Avg Price']
        
        # Product recommendations
        recommendations = {}
        for product_id in product_metrics.index:
            metrics = product_metrics.loc[product_id]
            
            # Scoring system
            revenue_score = min(metrics['Total Revenue'] / product_metrics['Total Revenue'].max(), 1.0)
            profit_score = min(metrics['Total Profit'] / product_metrics['Total Profit'].max(), 1.0)
            engagement_score = min(metrics['Avg Page Views'] / product_metrics['Avg Page Views'].max(), 1.0)
            
            overall_score = (revenue_score * 0.4 + profit_score * 0.4 + engagement_score * 0.2)
            
            if overall_score >= 0.7:
                action = "PROMOTE"
                reason = "High performance across all metrics"
            elif overall_score >= 0.4:
                action = "KEEP"
                reason = "Stable performance"
            else:
                action = "REVIEW"
                reason = "Low performance - consider optimization or discontinuation"
            
            recommendations[product_id] = {
                'action': action,
                'reason': reason,
                'score': round(overall_score, 2)
            }
        
        results['product_metrics'] = product_metrics
        results['recommendations'] = recommendations
        results['processed_data'] = processed_data
        
        return results

# Streamlit App
def main():
    st.title("Product Rationalization")
    
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Data loading
    if data_source == "Generate Synthetic Data":
        num_records = st.sidebar.slider("Number of Records", 100, 2000, 1000)
        if st.sidebar.button("Generate Data"):
            with st.spinner("Generating synthetic data..."):
                generator = DataGenerator()
                st.session_state.data = generator.generate_synthetic_data(num_records)
                st.success(f"Generated {num_records} records!")
    
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
    
    # Main analysis
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Display data overview
        st.header("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Products", data['product_id'].nunique() if 'product_id' in data.columns else 'N/A')
        with col3:
            st.metric("Date Range", f"{len(pd.to_datetime(data.iloc[:, 0], errors='coerce').dropna().dt.date.unique())} days")
        with col4:
            revenue_col = next((col for col in data.columns if 'revenue' in col.lower() or 'sales' in col.lower()), None)
            if revenue_col:
                st.metric("Total Revenue", f"${data[revenue_col].sum():,.0f}")
            else:
                st.metric("Total Revenue", "N/A")
        
        # Data preview
        with st.expander("View Raw Data"):
            st.dataframe(data.head(20))
        
        # Run analysis
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                engine = ProductRationalizationEngine()
                st.session_state.analysis_results = engine.analyze_products(data)
            st.success("Analysis completed!")
        
        # Display results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            processed_data = results['processed_data']
            
            # Product Performance Dashboard
            st.header("Product Performance Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Product Recommendations")
                for product_id, rec in results['recommendations'].items():
                    st.write(f"**{product_id}**: {rec['action']}")
                    st.write(f"*{rec['reason']} (Score: {rec['score']})*")
                    st.write("---")
            
            with col2:
                st.subheader("Performance Metrics")
                st.dataframe(results['product_metrics'])
            
            # Visualizations
            st.header("Analytics Dashboard")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Sales Trends", "Price Optimization", "Forecasting", "Insights"])
            
            with tab1:
                # Sales trend chart
                daily_sales = processed_data.groupby('Date')['sales_qty'].sum().reset_index()
                fig = px.line(daily_sales, x='Date', y='sales_qty', title='Daily Sales Trend')
                st.plotly_chart(fig, use_container_width=True)
                
                # Category performance
                category_perf = processed_data.groupby('category')['revenue'].sum().reset_index()
                fig2 = px.bar(category_perf, x='category', y='revenue', title='Revenue by Category')
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                st.subheader("Price Optimization")
                
                # Price optimization for each product
                price_optimizer = PriceOptimizer(processed_data)
                
                for product_id in processed_data['product_id'].unique()[:5]:  # Limit to first 5 products
                    current_price = processed_data[processed_data['product_id'] == product_id]['price'].mean()
                    optimal_price, details = price_optimizer.optimize_price(product_id, current_price)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{product_id} - Current", f"${current_price:.2f}")
                    with col2:
                        st.metric("Optimal", f"${optimal_price:.2f}")
                    with col3:
                        change_pct = ((optimal_price - current_price) / current_price) * 100
                        st.metric("Change", f"{change_pct:+.1f}%")
                    
                    st.caption(details)
                    st.write("---")
            
            with tab3:
                st.subheader("Forecasting")
                
                forecast_days = st.slider("Forecast Days", 7, 90, 30)
                forecast_method = st.selectbox("Method", ["LSTM", "ARIMA", "Simple"])
                
                if st.button("Generate Forecast"):
                    with st.spinner("Generating forecast..."):
                        forecasting_engine = ForecastingEngine()
                        
                        if forecast_method == "LSTM":
                            forecast = forecasting_engine.forecast_lstm(processed_data, forecast_days)
                        elif forecast_method == "ARIMA":
                            forecast = forecasting_engine.forecast_arima(processed_data, forecast_days)
                        else:
                            forecast = forecasting_engine.simple_forecast(processed_data, forecast_days)
                        
                        # Create forecast visualization
                        last_date = processed_data['Date'].max()
                        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
                        
                        # Historical data
                        historical = processed_data.groupby('Date')['sales_qty'].sum().reset_index()
                        
                        # Combine historical and forecast
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=historical['Date'], 
                            y=historical['sales_qty'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast,
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(title=f'{forecast_method} Forecast - Next {forecast_days} Days')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        st.write(f"**Forecast Summary ({forecast_method})**")
                        st.write(f"- Average daily demand: {forecast.mean():.1f} units")
                        st.write(f"- Total forecasted demand: {forecast.sum():.0f} units")
                        st.write(f"- Peak demand day: Day {np.argmax(forecast) + 1} ({forecast.max():.1f} units)")
            
            with tab4:
                st.subheader("AI-Generated Insights")
                
                # Generate insights
                insight_generator = LLMInsightGenerator()
                insights = insight_generator.generate_insights(processed_data, results)
                recommendations = insight_generator.generate_recommendations(results)
                
                st.write("### Business Insights")
                for insight in insights:
                    st.markdown(insight)
                
                st.write("### Recommendations")
                for rec in recommendations:
                    st.markdown(rec)
                
                
        
        # Export functionality
        st.header("Export Results")
        if st.session_state.analysis_results:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Analysis Report"):
                    # Create comprehensive report
                    report_data = {
                        'Product_Metrics': results['product_metrics'],
                        'Recommendations': pd.DataFrame(results['recommendations']).T,
                        'Processed_Data': processed_data
                    }
                    
                    # Convert to Excel (simplified version for demo)
                    csv_data = results['product_metrics'].to_csv()
                    st.download_button(
                        "Download Product Metrics CSV",
                        csv_data,
                        "product_analysis.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("Export Forecast Data"):
                    # Generate a simple forecast for export
                    forecasting_engine = ForecastingEngine()
                    forecast = forecasting_engine.simple_forecast(processed_data, 30)
                    
                    forecast_df = pd.DataFrame({
                        'Date': pd.date_range(start=processed_data['Date'].max() + timedelta(days=1), periods=30),
                        'Forecasted_Sales': forecast
                    })
                    
                    csv_forecast = forecast_df.to_csv(index=False)
                    st.download_button(
                        "Download Forecast CSV",
                        csv_forecast,
                        "sales_forecast.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()