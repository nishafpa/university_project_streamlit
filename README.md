# Project Title

UC Berkely Professional Certificate for ML &amp; AI - Capstone Project: US Equity Closing Price Prediction

## Description

This project is to predict the Equity closing price of the US stock market listed equities based on the historical data such as Endogenous and Exogenous variables (Features).

Endogenous variables of the equity closing price prediction are Open, Low, High, Close/Adjusted Close and Volume from Yahoo Finance, and calculated values for Simple Moving Average (SMA), Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD) and Relative Strength Index (RSI).

Exogenous variables of the equity closing price prediction are Benchmark Indices (S&P500 Index, NASDAQ Composite Index) from Yahoo Finance and Macroeconomic indicators from Federal Reserve Economic Data (FRED) database API (Interest Rates & Monetary Policy, Yield Curve & Bond Market Signals, Credit Spreads & Financial Stress, Liquidity & Money Supply and Inflation & Macro Growth).
  
The classical Time Series Analysis & Forecasting and LSTM (Long Short-Term Memory) in Recurrent Neural Networks (RNN) are used to predict the US Equity price. These models are compared against the Baseline Model Linear Regression, to understand the actual models beat the baseline model.

## Getting Started

### Dependencies

* This Python project references the Libraries - numpy, pandas, matplotlib, seaborn, yfinance, fredapi and scikit-learn

### Installing

* Using the below command, install the required Libraries
* %pip install numpy pandas matplotlib seaborn yfinance fredapi scikit-learn
* For the convenience, it is included as the first cell in the Python Code

### Executing program

* Click the below URL to Navigate to the Code, to Download and to Execute on any Python Notebook Cell by Cell from Top to Bottom
* [US Equity Closing Price Prediction](https://github.com/nishafpa/university_project_001/blob/main/EquityClosingPricePrediction.ipynb)

## Help

Brief Explanation and Analysis are provided below each cell, as Cell Output and Markdown

## Author

Nishaf P Abdul Majeed<BR>
Senior Data Architect/Modeler<BR>
[nishafpa@gmail.com](mailto:nishafpa@gmail.com)<BR>
[LinkedIn](https://www.linkedin.com/in/nishaf-abdul-majeed-04b13ba3/)

## Version History

* 0.1 - Initial Release

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

UC Berkely Tutors for this Professional Course in Machine Learning and Artificial Intelligence

## Brief Report to Trader or Equity Fund Manager

When a Trader or Equity Fund Manager wants to invest in a particular Equity or the Equity Fund on behalf of their firm or their clients, the Prediction of the future closing price would help to make their decision. Without a Quantitative analysis, the decision would be based on guess work and it can potentially end up in wrong decision and can potentially lose the money. Quantitative analysis would help to decrease or minimize the money loss for the firm or their clients. But, be careful that even the best models cannot predict 100% accuracy, but can minimize the loss.