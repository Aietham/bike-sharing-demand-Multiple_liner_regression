# Bike Demand Prediction Using Multiple Linear Regression

## 1. Introduction
BoomBikes, a U.S. bike-sharing provider, aims to forecast daily demand for shared bicycles in the aftermath of the COVID-19 lockdown. This project leverages historical usage and weather-related features to build a multiple linear regression model predicting total daily rentals (`cnt`). The insights will help BoomBikes optimize inventory, pricing, and station placement.

## 2. Problem Statement
- **Goal:** Identify significant factors influencing bike demand and quantify how well they explain rental counts.
- **Target Variable:** `cnt` (total number of daily bike rentals, including both casual and registered users).
- **Key Questions:**
  1. Which features significantly predict bike demand?
  2. How accurately can we model `cnt` using a linear regression approach?

## 3. Data Description

| File                       | Description                                                                   |
|----------------------------|-------------------------------------------------------------------------------|
| `bike_sharing.csv`         | Daily rental counts with features: season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt |
| `data_dictionary.csv`      | Definitions and coding for categorical variables (season, weathersit, yr, etc.) |

## 4. Data Preparation
1. **Load and inspect** raw data (missing values, data types).
2. **Convert ordinal codes to categories**:
   - `season`: {1=Spring, 2=Summer, 3=Fall, 4=Winter}
   - `weathersit`: {1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain/Snow}
3. **Retain** `yr` (0=2018, 1=2019) to capture yearly growth trend.
4. **Drop** `casual` and `registered` (components of `cnt`) to prevent leakage.
5. **Train/test split**: typically 70/30 or 80/20 based on `mnth` or random shuffle.

## 5. Model Building
1. **Feature encoding**: One-hot encode categorical variables (`season`, `weathersit`, `weekday`, etc.).
2. **Standardization**: Scale numerical predictors (`temp`, `atemp`, `hum`, `windspeed`) as needed.
3. **Instantiate & train** a `LinearRegression` model from scikit-learn.
4. **Cross-validation** (optional): Evaluate stability with k-fold CV.

## 6. Model Evaluation
- Generate predictions on the test set.
- Compute **R-squared** using:
  ```python
  from sklearn.metrics import r2_score
  r2_score(y_test, y_pred)
  ```
- Analyze residuals for patterns (e.g., plot predicted vs. actual, residual histogram).


## 7. Environment & Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/<username>/Bike-Demand-Regression.git
   cd Bike-Demand-Regression
   ```
2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 8. Usage
1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/bike_demand_lr.ipynb
   ```
2. **Follow sections**:
   - EDA and missing-value handling
   - Categorical conversion & feature engineering
   - Model training & evaluation
   - Residual analysis
3. **View results** and interpret model performance.

## 9. Future Work
- Explore nonlinear models (e.g., decision trees, GBMs).
- Incorporate external data (holiday calendars, special events).
- Build a dashboard for real-time demand forecasting.

