# Import the Libraries
import streamlit as st
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sympy import true
import yfinance as yf
from fredapi import Fred

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
# Please note that you need to sign up for a free FRED account and obtain an API key from the FRED website (https://fred.stlouisfed.org/) to access the data
# You can replace your FRED API key in the code below OR use my provided API key for demonstration purpose
FRED_API_KEY = "c0f88df6d55d87792b278b64733d9b24"

# Header Display
def show_header():
    # Page Config & Title
    st.set_page_config(page_title="US Equity Closing Price Prediction", layout="wide")
    st.title("UC Berkely Professional Certificate for ML & AI - Capstone Project: US Equity Closing Price Prediction")

    st.subheader("Description")
    st.write(f"This project is to predict the Equity closing price of the US stock market listed equities based on the historical data such as Endogenous and Exogenous variables (Features).")
    st.write(f"Endogenous variables of the equity closing price prediction are Open, Low, High, Close/Adjusted Close and Volume from Yahoo Finance, and calculated values for Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD) and Relative Strength Index (RSI).")
    st.write(f"Exogenous variables of the equity closing price prediction are Benchmark Indices (S&P500 Index, NASDAQ Composite Index) from Yahoo Finance and Macroeconomic indicators from Federal Reserve Economic Data (FRED) database API (Interest Rates & Monetary Policy, Yield Curve & Bond Market Signals, Credit Spreads & Financial Stress, Liquidity & Money Supply and Inflation & Macro Growth).")
    st.write(f"The classical Time Series Analysis & Forecasting and LSTM (Long Short-Term Memory) in Recurrent Neural Networks (RNN) are used to predict the US Equity price. These models are compared against the Baseline Model Linear Regression, to understand the actual models beat the baseline model.")

    st.subheader("Brief Report to the Trader or Equity Fund Manager")
    st.write(f"When a Trader or Equity Fund Manager wants to invest in a particular Equity or the Equity Fund on behalf of their firm or their clients, the Prediction of the future closing price would help to make their decision. Without a Quantitative analysis, the decision would be based on guess work and it can potentially end up in wrong decision and can potentially lose the money. Quantitative analysis would help to decrease or minimize the money loss for the firm or their clients. But, be careful that even the best models cannot predict 100% accuracy, but can minimize the loss.")
    st.divider() # The separator
    st.info("💡 Please Enter a US Equity Ticker Symbol in the sidebar to proceed!")

# Footer Display
def show_footer():
    st.divider() # The separator
    st.markdown(f"Author: Nishaf P Abdul Majeed (Senior Data Architect/Modeler) | [nishafpa@gmail.com](mailto:nishafpa@gmail.com) | [LinkedIn](https://www.linkedin.com/in/nishaf-abdul-majeed-04b13ba3/)")
    st.write(f"Acknowledgements: UC Berkely Tutors for this Professional Course in Machine Learning and Artificial Intelligence!")
    st.write(f"Copyright (c) 2026 Nishaf P Abdul Majeed. All rights reserved.")
    st.divider() # The separator

# Validate the ticker symbol using yfinance
def ticker_validation(ticker: str) -> dict:
    try:
        if ticker == "":
            return {"status": False, "message": "ERROR: Empty US Equity Ticker Symbol entered!"}
        elif 'regularMarketPrice' in yf.Ticker(ticker).info:
            return {"status": True, "message": ""}
        else:
            return {"status": False, "message": f"ERROR: Invalid US Equity Ticker Symbol {ticker} entered or Yahoo Finance does not return the data requested!"}
    except Exception as e:
        return {"status": False, "message": f"ERROR: Invalid US Equity Ticker Symbol {ticker} entered or Yahoo Finance does not return the data requested!"}

def download_historical_equity_data(ticker: str) -> dict:
    try:
        # Download historical data for a single Equity from Yahoo Finance
        # The Adjusted Closing Price is generally more important for long term stock prediction, since it accounts for Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Adjusted Close Price, with the column name 'Close'
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=True)

        # Check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Rename the columns to lowercase and prefix with 'equity_'
            df.columns = ['equity_'+col.lower() for col in df.columns]

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for {ticker}!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for {ticker}!"}

def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    try:
        # Calculate the Technical Indicators for historical data of a single Equity

        # Calculate Simple Moving Average (SMA)
        # It is used to Identify Trend Direction, Support and Resistance Levels, and Generate Buy/Sell Signals
        # An SMA is an arithmetic moving average calculated by adding recent prices and then dividing that figure by the number of time periods in the calculation average
        # Short Term Price Movement - 20 days SMA
        # Medium Term Price Movement - 50 days SMA
        # Long Term Price Movement - 200 days SMA
        df['equity_sma_20'] = df['equity_close'].rolling(window=20).mean()
        df['equity_sma_50'] = df['equity_close'].rolling(window=50).mean()
        df['equity_sma_200'] = df['equity_close'].rolling(window=200).mean()

        # Calculate Exponential Moving Average (EMA)
        # It is used to Identify Fast Trend Direction, Support and Resistance Levels, and Generate Buy/Sell Signals
        # An EMA is a type of moving average that places a greater weight and significance on the most recent price data.
        # The EMA reacts more significantly to recent price changes than a simple moving average (SMA), which applies an equal weight to all observations in the period.
        # adjust=False ensures recursive calculation matching standard financial formula.
        # EMA = ((Current Closing Price - Previous EMA) * multiplier) + Previous EMA where multiplier = 2 / (span + 1)
        # span is the number of periods for the EMA calculation
        # Short Term Price Movement - 9 days EMA
        # Medium Term Price Movement - 50 days EMA
        # Long Term Price Movement - 100 days EMA
        df['equity_ema_9'] = df['equity_close'].ewm(span=9, adjust=False).mean()
        df['equity_ema_50'] = df['equity_close'].ewm(span=50, adjust=False).mean()
        df['equity_ema_100'] = df['equity_close'].ewm(span=100, adjust=False).mean()

        # Calculate Moving Average Convergence Divergence (MACD)
        # It is used to Identify Trend Direction, Momentum, and Generate Buy/Sell Signals
        exp12 = df['equity_close'].ewm(span=12, adjust=False).mean()
        exp26 = df['equity_close'].ewm(span=26, adjust=False).mean()
        df['equity_macd_line'] = exp12 - exp26 #  12-period EMA minus 26-period EMA
        df['equity_macd_signal'] = df['equity_macd_line'].ewm(span=9, adjust=False).mean() # 9-period EMA of the MACD line
        df['equity_macd_histogram'] = df['equity_macd_line'] - df['equity_macd_signal'] # MACD Histogram = MACD Line - Signal Line
        # Drop the intermediate variables to save memory
        del exp12, exp26
        # Drop MACD Line and MACD Signal columns, as they are intermediate calculations
        df.drop(columns=['equity_macd_line', 'equity_macd_signal'], inplace=True)

        # Calculate Relative Strength Index (RSI)
        # It is used to Identify Overbought and Oversold Conditions, Trend Strength, and Generate Buy/Sell Signals
        delta = df['equity_close'].diff() # Calculate the difference between the current and previous closing price
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean() # Calculate the average gain over a 14-day period
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() # Calculate the average loss over a 14-day period
        rs = gain / loss # Calculate the relative strength (RS) as the ratio of average gain to average loss
        df['equity_rsi_14'] = 100 - (100 / (1 + rs)) # Calculate the RSI using the formula: RSI = 100 - (100 / (1 + RS))
        # Drop the intermediate variables to save memory
        del delta, gain, loss, rs

        return {"data": df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Error occurred while calculating technical indicators!"}

def download_historical_sp500_data() -> dict:
    try:
        # Download historical data for S&P 500 Index (as a benchmark) from Yahoo Finance
        # S&P 500 Index Ticker Symbol: ^GSPC
        # For Index Tickers, Closing Price and Adjusted Closing Price are the same, since Indexes do not have Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Close Price, since Adjusted Close Price is the same as Close Price for Indexes
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download('^GSPC', period="max", interval="1d", auto_adjust=True)

        # check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Drop Columns that are not needed for analysis (e.g., 'Open', 'High', 'Low', 'Volume')
            df = df.drop(columns=['Open', 'High', 'Low', 'Volume'])
            # Rename the columns to lowercase and prefix with 'sp500_'
            df.columns = ['sp500_'+col.lower() for col in df.columns]

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for S&P 500 Index!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for S&P 500 Index!"}

def download_historical_nasdaq_data() -> dict:
    try:
        # Download historical data for NASDAQ Composite Index (as a benchmark) from Yahoo Finance
        # NASDAQ Composite Index Ticker Symbol: ^IXIC
        # For Index Tickers, Closing Price and Adjusted Closing Price are the same, since Indexes do not have Corporate Actions like Stock Splits, Dividends, and Rights Offerings
        # auto_adjust=True will return only Close Price, since Adjusted Close Price is the same as Close Price for Indexes
        # period="max" will download the maximum available data, interval="1d" will download daily data
        df = yf.download('^IXIC', period="max", interval="1d", auto_adjust=True)

        # check if the dataframe is empty (i.e., no data was returned)
        if len(df) > 0:
            # Clean multi-level column index if present in newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Drop Columns that are not needed for analysis (e.g., 'Open', 'High', 'Low', 'Volume')
            df = df.drop(columns=['Open', 'High', 'Low', 'Volume'])
            # Rename the columns to lowercase and prefix with 'nasdaq_'
            df.columns = ['nasdaq_'+col.lower() for col in df.columns]

            return {"data": df, "status": True, "message": ""}
        else:
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for NASDAQ Composite Index!"}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Yahoo Finance does not return the data requested for NASDAQ Composite Index!"}

def download_historical_fred_data(api_key: str) -> dict:
    try:
        # Download historical data from FRED (Federal Reserve Economic Data) database for the below:

        # Initialize the FRED object with API key
        # Please note that you need to sign up for a free FRED account and obtain an API key from the FRED website (https://fred.stlouisfed.org/) to access the data
        # You can replace your FRED API key in the code below OR use my provided API key for demonstration purpose
        fred = Fred(api_key=api_key)

        # Define the dictionary of series IDs
        fred_series_dict = {
            # Interest Rates & Monetary Policy - Central bank decisions drive equity discount rates and corporate borrowing costs
            'effective_federal_reserve_funds_rate': 'FEDFUNDS', # Monthly. The primary benchmark rate set by the Fed
            # Yield Curve & Bond Market Signals - The bond market often prices economic reality faster than equity markets
            'treasury_constant_maturity_10_year_minus_2_year': 'T10Y2Y', # Daily. The classic yield curve inversion indicator. When it dips below 0, it signals a looming recession, traditionally acting as a strong mid-to-long term bearish flag for equities
            'treasury_constant_maturity_10_year_minus_3_month': 'T10Y3M', # Daily. The Federal Reserve Bank of New York uses this specific 10-year minus 3-month spread to calculate the statistical probability of a U.S. recession over the next 12 months
            '10_year_treasury_constant_maturity_rate': 'DGS10', # Daily. Used as the risk-free rate in discounted cash flow (DCF) stock valuation models. A rising 10Y yield compresses equity valuation multiples
            # Credit Spreads & Financial Stress - These indicators serve as proxies for systemic risk and corporate bankruptcy risk
            'st_louis_fed_financial_stress_index': 'STLFSI4', # Weekly. A composite index measuring market stress. Readings above 0 indicate above-average financial market strain
            'cboe_volatility_index': 'VIXCLS', #  Daily. The "fear gauge". While technically a market output, incorporating it helps predict short-term regime changes and momentum shifts
            # Liquidity & Money Supply - Stocks require capital flow to move higher. These indicators track macro liquidity
            'm2_money_supply': 'M2SL', # Monthly. Measures cash, checking deposits, and easily convertible money. Rapid M2 growth historically signals asset price inflation (bullish for stocks)
            # Inflation & Macro Growth - These indicators affect consumer spending power and corporate margins
            'consumer_price_index': 'CPIAUCSL', # Monthly. The headline inflation indicator. Persistent high inflation forces aggressive Fed tightening, squeezing stock multiples
            'unemployment_rate': 'UNRATE', # Monthly. A lagging but vital health indicator of the consumer economy
            'industrial_production_index': 'INDPRO', # Monthly. A reliable, monthly proxy for GDP growth before official quarterly numbers are released
            'real_gross_domestic_product': 'GDPC1' # Quarterly. is generally more important for long-term stock market trends because it reflects true economic health and corporate volume growth without price distortions
        }

        # Download each series into a DataFrame and combine them
        fred_data_frames = []
        for name, series_id in fred_series_dict.items():
            try:
                series_data = fred.get_series(series_id)
                df_temp = pd.DataFrame(series_data, columns=[name])
                fred_data_frames.append(df_temp)
            except Exception as e:
                return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: FRED database does not return the data requested for {name}! Please check the series ID and your FRED API key!"}

        # Concatenate all series horizontally into a single DataFrame
        fred_original_df = pd.concat(fred_data_frames, axis=1)

        # Drop the intermediate variables to save memory
        del fred_data_frames

        return {"data": fred_original_df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: FRED database does not return the data! Please check the series ID and your FRED API key!"}

def data_preparation(ticker_df: pd.DataFrame, sp500_df: pd.DataFrame, nasdaq_df: pd.DataFrame, fred_df: pd.DataFrame) -> dict:
    try:
        # Get the data from 1/1/2021 from ticker_df dataframe for Post Covid analysis
        ticker_df_copy = ticker_df.copy()
        ticker_df_copy = ticker_df_copy[ticker_df_copy.index >= '2021-01-01']

        # Get the data from 1/1/2021 from sp500_df dataframe for Post Covid analysis
        sp500_df_copy = sp500_df.copy()
        sp500_df_copy = sp500_df_copy[sp500_df_copy.index >= '2021-01-01']

        # Get the data from 1/1/2021 from nasdaq_df dataframe for Post Covid analysis
        nasdaq_df_copy = nasdaq_df.copy()
        nasdaq_df_copy = nasdaq_df_copy[nasdaq_df_copy.index >= '2021-01-01']

        # Forward fill FRED DataFrame, since some series are Weekly, Monthly or Quarterly, and this will ensure that the data is aligned with the equity data downloaded from Yahoo Finance
        # Forward fill is used to propagate the last valid observation forward to the next valid observation, which is useful for time series data where we want to fill in missing values with the most recent available data point.
        fred_filled_df = fred_df.ffill()

        # Get the data from 1/1/2021 from fred_filled_df dataframe for Post Covid analysis
        fred_filled_df = fred_filled_df[fred_filled_df.index >= '2021-01-01']

        # Inner Join the dataframe ticker_df with sp500_df, nasdaq_df and fred_filled_df to create a merged dataframe
        df = ticker_df_copy.join([sp500_df_copy, nasdaq_df_copy, fred_filled_df], how="inner")

        # Create target column for next trading day's closing price in the current row, which will be used for supervised learning
        df['equity_predicted_close'] = df['equity_close'].shift(-1)

        # Drops the last row since it will have NaN value in the target column 'equity_predicted_close' after shifting
        df.dropna(inplace=True)

        # Get the number of missing values in each column
        # There should not be any missing values in the merged dataframe, since we have forward filled the FRED data and dropped the last row after shifting the target column
        if (df.isnull().sum().sum() > 0):
            return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Data Quality Check Failed since the Data contains Missing Values!"}

        return {"data": df, "status": True, "message": ""}
    except Exception as e:
        return {"data": pd.DataFrame(), "status": False, "message": f"ERROR: Data Preparation and/or Data Quality Check Failed!"}

def data_plot(df: pd.DataFrame) -> dict:
    try:
        # Plot the time-series data for all the features in the merged dataframe using Seaborn's lineplot function
        sns.set(style="darkgrid") # Set the Seaborn style to "darkgrid" for better aesthetics in the plots

        # Get the list of features (columns) in the dataframe, except the target column 'equity_predicted_close'
        features_list = [col for col in df.columns if col != 'equity_predicted_close']

        # Create a grid of subplots
        fig, axes = plt.subplots(
            nrows=len(features_list), # Number of rows equal to the number of features
            ncols=1, # Number of columns is 1 since we want a single column of plots
            figsize=(15, 100), # Set the figure size to 15 inches wide and 100 inches tall to accommodate all the subplots
            sharex=True,  # Share x-axis across all time-series panels
        )

        # Flatten axes array in case it's multi-dimensional, loop through each feature
        for i, feature in enumerate(features_list):
            ax = axes[i] # Get the current axis for the subplot
            sns.lineplot(data=df, x=df.index, y=feature, ax=ax, color="tab:blue") # Plot the time-series data for the current feature using Seaborn's lineplot function, with the x-axis as the index (date) and y-axis as the feature values, and set the color to blue
            ax.set_title(f"Feature: {feature}", loc="left", fontsize=12) # Set the title of the subplot to indicate the feature being plotted, aligned to the left, with a font size of 12
            ax.set_ylabel(feature) # Set the y-axis label to the feature name to indicate what is being plotted on the y-axis
            ax.set_xlabel("") # Set the x-axis label to an empty string to avoid cluttering the plots with x-axis labels for each subplot

        axes[-1].set_xlabel("Trade Date") # Label the bottom x-axis
        plt.tight_layout() # Adjust the layout to prevent overlapping of subplots and ensure that titles and labels are clearly visible
        plt.show() # Display the plots

        return {"fig": fig, "status": True, "message": ""}
    except Exception as e:
        return {"fig": fig, "status": False, "message": f"ERROR: Data Plotting Failed!"}

def linear_regression_model(df: pd.DataFrame) -> dict:
    try:
        # Create a Baseline Linear Regression Model

        # Create X and y from the equity_df dataset where 'equity_predicted_close' column is the target and the remaining columns are the features
        X = df.drop(columns=['equity_predicted_close']) # Data Frame for Features
        y = df["equity_predicted_close"] # Series for Target

        # Split the time series data into training and testing sets
        # Since this is time series data, we will not shuffle the data and will use the first 80% of the data for training and the last 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Create a Pipeline with Scaling and Linear Regression
        pipeline = Pipeline(steps=[
                ("scaler", MinMaxScaler()),  # Scaling is important to ensure that all features are on the same scale
                ("regressor", LinearRegression())
            ])

        # Define the Parameter Grid
        # The parameter grid is used to specify the hyperparameters that we want to tune for the Linear Regression model.
        # In this case, we are tuning the 'fit_intercept' and 'positive' parameters of the Linear Regression model.
        # fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g., data is expected to be already centered).
        # positive: When set to True, forces the coefficients to be positive. This can be useful when we know that the relationship between the features and the target variable is positive (e.g., as the value of a feature increases, the value of the target variable also increases).
        param_grid = {"regressor__fit_intercept": [True, False], "regressor__positive": [True, False]}

        # Time Series Split which is essential for financial data to prevent looking ahead into future data
        tscv = TimeSeriesSplit(n_splits=5)

        # Create a Grid Search with Cross-Validation
        # Cross-Validation is used to avoid data leakage and overfitting. The number of splits is set to 5, which means that the dataset will be split into 5 folds
        # Scoring is set to "neg_mean_squared_error" since we are predicting a continuous variable (equity closing price)
        # n_jobs=-1 means that all available CPU cores will be used for parallel processing, which can speed up the grid search process.
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)

        # Fit the Grid Search to the training data
        grid_search.fit(X_train, y_train)

        # Predict using the Best Model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Evaluate Metrics
        mae = round(mean_absolute_error(y_test, y_pred),4) # Best Value is 0. Measures the average absolute distance between actual and predicted values. It treats all errors equally.
        mse = round(mean_squared_error(y_test, y_pred),4) # Best Value is 0. Measures the average squared distance between actual and predicted values. It penalizes larger errors more than smaller errors.
        rmse = round(np.sqrt(mse),4) # Best Value is 0. Measures the square root of the average squared distance between actual and predicted values. It is in the same unit as the target variable and is more interpretable than MSE.
        r2 = round(r2_score(y_test, y_pred),4) # Best Value is 1. Measures the proportion of variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better model performance.

        # Plot the Actual vs Predicted Closing Price for the Test Set
        # Create DatetimeIndex object to store the index of the test set
        test_data_DatetimeIndex = X_test.index

        # Create a DataFrame in Wide Format for Seaborn Line Plot
        df_wide = pd.DataFrame({'Actual Closing Price': y_test, 'Predicted Closing Price': y_pred}, index=test_data_DatetimeIndex)

        # Define a custom color palette for the line plot to make it visually appealing and easy to distinguish between actual and predicted values
        custom_palette = {
            'Actual Closing Price': '#0A2540',    # Deep Navy Blue
            'Predicted Closing Price': '#00D4B2'  # Bright Teal
        }

        prediction_fig = plt.figure(figsize=(15, 5)) # Set the figure size to 15 inches wide and 5 inches tall
        sns.lineplot(data=df_wide, palette=custom_palette, linewidth=2.5) # Plot the actual and predicted closing prices using Seaborn's lineplot function, with the specified color palette and line width
        plt.title(f'Linear Regression - Actual vs Predicted Equity Closing Price') # Set the title of the plot to include the ticker symbol for clarity
        plt.xlabel('Trade Date') # Set the x-axis label to 'Trade Date' to indicate that the x-axis represents the dates of the trades
        plt.ylabel('Closing Price in USD') # Set the y-axis label to 'Closing Price' to indicate that the y-axis represents the closing prices of the equity
        plt.xticks(rotation=45) # Rotate the x-axis labels by 45 degrees for better readability, especially when there are many dates on the x-axis
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        # Plot the most influential features in the top using Seaborn
        # Create dataframe for Features and Coefficients
        feature_coefficient_df = pd.DataFrame({
            'Feature': [name.split('__')[-1] for name in list(best_model.named_steps['scaler'].get_feature_names_out())],
            'Coefficient': best_model.named_steps['regressor'].coef_
        })

        feature_fig = plt.figure(figsize=(15, 8)) # Set the figure size to 15 inches wide and 10 inches tall
        sns.barplot(data=feature_coefficient_df.sort_values(by='Coefficient', ascending=False), x='Coefficient', y='Feature', palette='viridis') # Plot the coefficients of the features using Seaborn's barplot function, sorting the features by their coefficient values in descending order, and using the 'viridis' color palette for better visual distinction
        plt.title(f'Linear Regression - Features most Influencing Equity Closing Price at the Top') # Set the title of the plot to indicate that it shows the features most influencing the equity closing price for the specified ticker
        plt.xlabel('Coefficient Value') # Set the x-axis label to 'Coefficient Value' to indicate that the x-axis represents the values of the coefficients for each feature
        plt.ylabel('Feature Name') # Set the y-axis label to 'Feature Name' to indicate that the y-axis represents the names of the features used in the model
        plt.tight_layout() # Adjust the padding of the plot to ensure that all elements are visible and not cut off
        plt.show() # Display the plot

        return {"metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": True, "message": ""}
    except Exception as e:
        return {"metrics": {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}, "prediction_fig": prediction_fig, "feature_fig": feature_fig, "status": False, "message": f"ERROR: Linear Regression Model Failed!"}

def main():
    show_header()

    # User inputs a US Equity Ticker Symbol
    with st.sidebar.form(key="ticker_form"):
        ticker = st.text_input("Please Enter a US Equity Ticker Symbol (e.g., AAPL):").strip().upper()
        submitted = st.form_submit_button(label="Submit")

    # This block waits until the submit button is clicked
    if submitted:
        return_dict = ticker_validation(ticker)
        if return_dict["status"]:
            st.write(f"##### US Equity Ticker Symbol: {ticker}")
        else:
            st.sidebar.error(return_dict['message'])
            show_footer()
            st.stop()

        st.write(f"##### Step# 1: Downloading the data from Yahoo Finance, FRED API and Calculating Technical Indicators...")
        download_progress_bar = st.progress(0, text="Progress: 0%")

        return_dict = download_historical_equity_data(ticker=ticker)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_equity_df = return_dict["data"]
        download_progress_bar.progress(20, text="Progress: 20%")
        st.write(f"Downloaded the data for {ticker} from Yahoo Finance!")

        return_dict = calculate_technical_indicators(df=yf_original_equity_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_equity_df = return_dict["data"]
        download_progress_bar.progress(30, text="Progress: 30%")
        st.write(f"Calculated the Technical Indicators for {ticker}!")

        return_dict = download_historical_sp500_data()
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_sp500_df = return_dict["data"]
        download_progress_bar.progress(40, text="Progress: 40%")
        st.write(f"Downloaded the data for S&P 500 Index from Yahoo Finance!")
        
        return_dict = download_historical_nasdaq_data()
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        yf_original_nasdaq_df = return_dict["data"]
        download_progress_bar.progress(50, text="Progress: 50%")
        st.write(f"Downloaded the data for NASDAQ Composite Index from Yahoo Finance!")

        return_dict = download_historical_fred_data(api_key=FRED_API_KEY)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        fred_original_df = return_dict["data"]
        download_progress_bar.progress(100, text="Progress: 100%")
        st.write(f"Downloaded the Economic Indicators data from FRED database!")

        st.write(f"##### Step# 2: Creating Final Merged dataset from 01/01/2021, Checking Data Quality, Preparting the Plots...")
        return_dict = data_preparation(ticker_df=yf_original_equity_df, sp500_df=yf_original_sp500_df, nasdaq_df=yf_original_nasdaq_df, fred_df=fred_original_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        equity_df = return_dict["data"]
        st.write(f"Final Dataset Created and Data Quality Check Passed!")
        st.dataframe(equity_df, use_container_width=True)
        
        return_dict = data_plot(df=equity_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.write(f"Dataset Plotted!")
        st.pyplot(return_dict["fig"])

        st.write(f"##### Step# 3: Creating the Baseline Model (Linear Regression), Predicting Next Day's Closing Price and Checking Evaluation Metrics...")
        return_dict = linear_regression_model(df=equity_df)
        if not return_dict["status"]:
            st.error(return_dict['message'])
            show_footer()
            st.stop()
        st.pyplot(return_dict["prediction_fig"])
        st.pyplot(return_dict["feature_fig"])
        # Print Metrics
        st.write(f"Linear Regression - Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {return_dict["metrics"]["mae"]} (Best Value is 0)")
        st.write(f"Mean Squared Error (MSE): {return_dict["metrics"]["mse"]} (Best Value is 0)")
        st.write(f"Root Mean Squared Error (RMSE): {return_dict["metrics"]["rmse"]} (Best Value is 0)")
        st.write(f"R-squared (R2): {return_dict["metrics"]["r2"]} (Best Value is 1)")

    show_footer()

if __name__ == "__main__":
    main()