
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import os

## cache the data
@st.cache_data(ttl=3600, show_spinner="........")
def load_data(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# https://gardnmi.github.io/blog/jupyter,/prophet,/stock,/python/2020/10/20/stock-price-forecast-with-prophet.html#Prophet

# st.set_page_config(layout='wide', initial_sidebar_state='expanded')
period = 3

st.title('Using Prophet for Trading')
st.write(f'''
         Compare trading algorithms to decide when to buy and sell for {period} years of data to see which can make us most money. We will use the google stock price as our example.
         ''')

if os.path.exists('./data/GOOG.csv'):
    data = pd.read_csv(r'./data/GOOG.csv', parse_dates=['Date'])
else:
# Read the training file into a DataFrame
    data = load_data('GOOG', '2013-01-01', '2023-12-31')
    data.to_csv(r'./data/GOOG.csv', index=False)  
    

st.title('2. Prophet')
st.write(f"""
         We saw in part 1 that Prophet is easy to use and is highly parameterizable! But how would we use it in practice and how well does it perform?
         From the wide y hat range of future predictions in the plot below, we need to use it in a clever way. 
         """)
# Predict forecast with Prophet.
stock_price = data[['Date','Close']]
stock_price = stock_price.rename(columns={"Date": "ds", "Close": "y"})

# instantiate the model and set parameters
model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)
model.add_country_holidays(country_name='US')
model.fit(stock_price)

future = model.make_future_dataframe(periods=period * 365, 
    freq='d', 
    include_history=True)

future_boolean = future['ds'].map(lambda x : True if x.weekday() in range(0, 5) else False)
future = future[future_boolean] 
forecast = model.predict(future)
GOOG_stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
df = pd.merge(stock_price, GOOG_stock_price_forecast, on='ds', how='right')
fig1, ax1 = plt.subplots(figsize=(10, 6))
df.set_index('ds').plot(figsize=(16,8), color=['royalblue', "#34495e", "#e74c3c", "#e74c3c"], grid=True, ax=ax1)
st.pyplot(fig1)


st. title('3. Using Prophet in a Smarter Way')
st.markdown(
'''
Perhaps we could make predictions for a smaller window into the future and continuously retrain ...  
In this section we will simulate and backtest Prophet to create a monthly forecast.  
I preprocess the code to create a monthly forecast. The data transformation in the background involves indexing the data by month/year.
''')
stock_price['dayname'] = stock_price['ds'].dt.day_name()
stock_price['month'] = stock_price['ds'].dt.month
stock_price['year'] = stock_price['ds'].dt.year
stock_price['month/year'] = stock_price['month'].map(str) + '/' + stock_price['year'].map(str) 
stock_price = pd.merge(stock_price, 
                stock_price['month/year'].drop_duplicates().reset_index(drop=True).reset_index(),
                on='month/year',
                how='left')
stock_price = stock_price.rename(columns={'index':'month/year_index'})
'''
We loop through each month/year in our stock prices dataframe and fit the Prophet model with the data available to that period and then forecasting out one month ahead until we hit the last unique month/year. 
The forecasts are combined into a single dataframe which I save as a CSV since training so many models on loop takes a while to run. 
'''

if os.path.exists('./data/GOOG_stock_price_forecast.csv'):
	GOOG_stock_price_forecast = pd.read_csv('./data/GOOG_stock_price_forecast.csv', parse_dates=['ds'])
else:
    loop_list = stock_price['month/year'].unique().tolist()
    max_num = len(loop_list) - 1
    forecast_frames = []
    for num, item in enumerate(loop_list):
        if  num == max_num:
            pass
        else:
            df = stock_price.set_index('ds')[
                stock_price[stock_price['month/year'] == loop_list[0]]['ds'].min():\
                stock_price[stock_price['month/year'] == item]['ds'].max()]

            df = df.reset_index()[['ds', 'y']]
            model = Prophet()

            model.fit(df)

            future = stock_price[stock_price['month/year_index'] == (num + 1)][['ds']]

            forecast = model.predict(future)
            forecast_frames.append(forecast)

    GOOG_stock_price_forecast = reduce(lambda top, bottom: pd.concat([top, bottom], sort=False), forecast_frames)
    GOOG_stock_price_forecast = GOOG_stock_price_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    GOOG_stock_price_forecast.to_csv('./data/GOOG_stock_price_forecast.csv', index=False)

'''
Let's examine the results. Comparing the actuals to our predictions, we see that the predictions behave like a delayed moving average
'''

df = pd.merge(stock_price[['ds','y', 'month/year_index']], GOOG_stock_price_forecast, on='ds')
df['Percent Change'] = df['y'].pct_change()
df.set_index('ds')[['y', 'yhat', 'yhat_lower', 'yhat_upper']].plot(figsize=(16,8), color=['royalblue', "#34495e", "#e74c3c", "#e74c3c"], grid=True)
st.pyplot(plt)

st.title('4. Trading Algorithms')
st.markdown('''Let us use this data to simulate how various trading strategies did over time with an initial investment of $1,000 dollars. 
We create four initial trading algorithms:
* **Hold**: Our bench mark.  This is a buy and hold strategy. Meaning we buy the stock and hold on to it until the end time period.
* **Prophet**: This strategy is to sell when our forecast indicates a down trend and buy back in when it iindicates an upward trend
* **Prophet Thresh**: This strategy is to only sell when the stock price fall below our yhat_lower boundary.
* **Seasonality**:  This strategy is to exit the market in August and re-enter in Ocober. This was based on the seasonality chart from above. 
''')
df['Hold'] = (df['Percent Change'] + 1).cumprod()
df['Prophet'] = ((df['yhat'].shift(-1) > df['yhat']).shift(1) * (df['Percent Change']) + 1).cumprod()
df['Prophet Thresh']  = ((df['y'] > df['yhat_lower']).shift(1)* (df['Percent Change']) + 1).cumprod()
df['Seasonality'] = ((~df['ds'].dt.month.isin([8,9])).shift(1) * (df['Percent Change']) + 1).cumprod()

fig3, ax3 = plt.subplots(figsize=(16,8))
(df.dropna().set_index('ds')[['Hold', 'Prophet', 'Prophet Thresh','Seasonality']] * 1000).plot(figsize=(16,8), grid=True, ax=ax3)
st.pyplot(fig3)


st.write(f"Hold = {df['Hold'].iloc[-1]*1000:,.0f}")
st.write(f"Prophet = {df['Prophet'].iloc[-1]*1000:,.0f}")
st.write(f"Prophet Thresh = {df['Prophet Thresh'].iloc[-1]*1000:,.0f}")
st.write(f"Seasonality = {df['Seasonality'].iloc[-1]*1000:,.0f}")

st.write("How delightful that holding outperforms Prophet in both of our methods. Seasonality does better but this was based on some degree of future knowledge")

st.title('5. Optimizing the Threshold')
''' 
In an attempt to improve how we use Prophet and beat our holding method, let's create an optimized threshold at each time point based on a rolling windows to incorporate changing thresholds over time.
We will use rolling windows of 10, 20, ... 96 months and and threshold as a proportion from 0 to 0.99"
'''
if os.path.exists('./data/GOOG_rolling_thresh.csv'):
    GOOG_rolling_thresh = pd.read_csv('./data/GOOG_rolling_thresh.csv')
else:

    GOOG_rolling_thresh = {}

    for num, index in enumerate(df['month/year_index'].unique()):
    
        rolling_performance = {}
        
        for roll in range(10, 96, 10):
                
            temp_df = df.set_index('ds')[
                df[df['month/year_index'] == index]['ds'].min() - pd.DateOffset(months=roll):\
                df[df['month/year_index'] == index]['ds'].max()]

            performance = {}
            
            for thresh in np.linspace(.0,.99, 100):
                percent =  ((temp_df['y'] > temp_df['yhat_lower'] * thresh).shift(1)* (temp_df['Percent Change']) + 1).cumprod()
                performance[thresh] = percent
                
            per_df =  pd.DataFrame(performance).apply(pd.to_numeric)
            best_thresh = per_df.iloc[-1].idxmax()
            percents = per_df[best_thresh]
            
            rolling_performance[best_thresh] = percents
        
        per_df =  pd.DataFrame(rolling_performance).apply(pd.to_numeric)
        best_GOOG_rolling_thresh = per_df.iloc[[-1]].max().idxmax()
        
        if num == len(df['month/year_index'].unique())-1:
            pass
        else:
            GOOG_rolling_thresh[df['month/year_index'].unique()[num+1]] = best_GOOG_rolling_thresh

    GOOG_rolling_thresh = pd.DataFrame([GOOG_rolling_thresh]).T.reset_index().rename(columns={'index':'month/year_index', 0:'Fcst Thresh'})
    GOOG_rolling_thresh.to_csv('./data/GOOG_rolling_thresh.csv', index=False)



# plot the optimized threshold
fig5, ax5 = plt.subplots(figsize=(16,8))
GOOG_rolling_thresh['Fcst Thresh'].plot(figsize=(16,8), grid=True, ax=ax5)
st.pyplot(fig5)




GOOG_rolling_thresh['Fcst Thresh'].plot(figsize=(16,8), grid=True)
df['yhat_optimized'] = pd.merge(df, GOOG_rolling_thresh, 
                                on='month/year_index', 
                                how='left')['Fcst Thresh'].fillna(1).shift(1) * df['yhat_lower']
df['Prophet Rolling Thresh']  = ((df['y'] > df['yhat_optimized']).shift(1)* (df['Percent Change']) + 1).cumprod()

fig4, ax4 = plt.subplots(figsize=(16,8))
(df.dropna().set_index('ds')[['Hold', 'Prophet', 'Prophet Thresh', 'Prophet Rolling Thresh', 'Seasonality']] * 1000).plot(figsize=(16,8), grid=True, ax=ax4)
st.pyplot(fig4)

st.write(f"Hold = {df['Hold'].iloc[-1]*1000:,.0f}")
st.write(f"Prophet = {df['Prophet'].iloc[-1]*1000:,.0f}")
st.write(f"Prophet Thresh = {df['Prophet Thresh'].iloc[-1]*1000:,.0f}")
st.write(f"Seasonality = {df['Seasonality'].iloc[-1]*1000:,.0f}")
st.write(f"Prophet Rolling Thresh = {df['Prophet Rolling Thresh'].iloc[-1]*1000:,.0f}")

st.write('Once again, holding outperforms Prophet. The optimized threshold does better than the fixed threshold but still does not beat holding. ')