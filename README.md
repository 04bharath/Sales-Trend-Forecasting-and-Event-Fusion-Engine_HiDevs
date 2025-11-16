# Sales Trend Forecasting and Event Fusion Engine
# Project Overview
AI Sales Predictor Pro is an enterprise-grade retail forecasting system that uses XGBoost machine learning to predict future sales by analyzing multiple factors including promotions, holidays, weather, and competitor pricing. The system provides natural language interaction through a Streamlit web interface, delivering data-driven insights for inventory optimization and campaign planning.

# Features
**Multi-factor Analysis:** Combines historical sales, promotions, holidays, weather, and competitor data.

**XGBoost Model**: Industry-standard machine learning for accurate forecasts.

**Confidence Intervals:** Shows prediction ranges, not just single numbers.

**Real-time Explanations:** Identifies key factors affecting sales.

# Natural Language Interface
**Text-based Queries:**  Ask questions in plain English

**Smart Parsing:** Automatically extracts stores, products, regions, and conditions

**Quick Templates:** One-click examples for common scenarios

# Business Intelligence
**Visual Forecasts:** Interactive charts with confidence bands

**Factor Importance:** Shows what drives sales changes

**Actionable Recommendations:** Data-driven business advice

**Scenario Analysis:** What-if predictions for different conditions

# Architecture
Data Sources → Historical Sales Data ,
XGBoost Model →  ML Training & Prediction,
Streamlit App → Web Interface +  NLP Parser,
Business Insights →  Predictions +  Recommendations,

# Data Format
**date :**	  Transaction date,
**store_id**	→ Store identifier,
**region**	→ Geographic region,
**product_id**  →	Product identifier,
**sales**  →	Units sold (target),
**price**  →	Product price,
**promo_flag**  →	Promotion indicator (0/1),
**promo_depth**  →	Discount percentage,
**holiday_flag** →	Holiday indicator (0/1),
**holiday_name** →	Holiday name,
**temp_c** →	Temperature in Celsius,
**precipitation_mm** →	Rainfall in mm,
**stock_available** →	Inventory count,
**competitor_price** →	Competitor's price,
