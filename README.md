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

### ⚙️ **Model Details**
- Trained using **XGBoost**  
- Pickle-based model loading  
- Feature extraction + label encoding  
- Auto-fallback safety defaults


## 🏗️ Architecture (High-Level)

The system has **4 main components**:

1. **User Interface (Streamlit)**
   - Accepts natural language queries  
   - Displays predictions, charts, and recommendations  

2. **NLP Query Engine**
   - Extracts store, product, discount, weather, holidays  
   - Converts user query → machine-readable entities  

3. **Feature Engineering Layer**
   - Converts entities into model features  
   - Handles encoding, ratios, flags, seasonality  
   - Ensures correct feature order for the ML model  

4. **Prediction & Analytics Engine (XGBoost)**
   - Loads trained model from `sales_model.pkl`  
   - Generates sales prediction + confidence interval  
   - Computes feature importance  
   - Produces insights & business recommendations  

All components work together to generate a complete AI-driven sales analysis.


## 📄 Data Format

The model uses a structured dataset where each row represents **one product’s sales record for a day**.

### Required Columns
- `store_id` – Store code (e.g., S01, S07)
- `product_id` – Product code (e.g., P014, P019)
- `region` – Region name (North, South, East)
- `price` – Selling price on that day
- `promo_flag` – 1 if discount/promotion is active
- `promo_depth` – Discount percent (0.20 = 20%)
- `holiday_flag` – 1 if the date is festival/holiday
- `temp_c` – Temperature of the day
- `precipitation_mm` – Rainfall in mm
- `competitor_price` – Competitor’s product price
- `stock_available` – Available stock units
- `month` – Month number (1–12)
- `day_of_week` – Day index (0=Mon)
- `is_weekend` – 1 for Saturday/Sunday
- `price_ratio` – price / competitor_price
- `promo_effectiveness` – promo_flag × promo_depth
- `sales` – Actual sales units (target variable)

  ## ⚙️ Setup Guide

Follow these steps to run the project on your system:

### 1️⃣ Install Python
Make sure Python 3.10+ is installed.

Check version:
```sh
python --version
# Install required libraries
pip install --upgrade pip
pip install -r requirements.txt

# Run the application
streamlit run app.py

# Project Structure
├── app.py                  # Main Streamlit application
├── sales_model.pkl         # Trained ML model (must be included)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
