
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

# https://gardnmi.github.io/blog/jupyter,/prophet,/stock,/python/2020/10/20/stock-price-forecast-with-prophet.html#Prophet

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.title('Stock Forecast App')
st.sidebar.header('Dashboard `version 1`')

st.sidebar.subheader('Stock parameter')
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.sidebar.selectbox('Select dataset for prediction', stocks)

# START = "2015-01-01"
START = st.sidebar.date_input('Input Start Date', value=date(2015, 1, 1), min_value=date(2015, 1, 1), max_value=date(2022, 1, 1), key='start_date')
TODAY = st.sidebar.date_input('Input End Date', value=date.today(), min_value=date(2015, 1, 1), max_value=date.today(), key='end_date')
START = START.strftime("%Y-%m-%d")
TODAY = TODAY.strftime("%Y-%m-%d")


st.sidebar.subheader('Stock Prediction parameter')
period = st.sidebar.slider('Years of prediction:', 1, 3)

st.sidebar.markdown('''
---
Created with ❤️ !
''')

## cache the data
@st.cache_data(ttl=3600, show_spinner="........")
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')


st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)
	

# Predict forecast with Prophet.
stock_price = data[['Date','Close']]
stock_price = stock_price.rename(columns={"Date": "ds", "Close": "y"})

# instantiate the model and set parameters
m = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
)
m.fit(stock_price)
future = m.make_future_dataframe(periods=period * 365, 
    freq='d', 
    include_history=True)

future_boolean = future['ds'].map(lambda x : True if x.weekday() in range(0, 5) else False)
future = future[future_boolean] 

forecast = m.predict(future)

# Show and plot forecast
st.subheader(f'Forecast {period} year of data')
st.write(forecast.tail())
'''The plot functions creates a graph of our actuals and forecast'''
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
'''plot_components provides us a graph of our trend and seasonality.'''
fig2 = m.plot_components(forecast)
st.write(fig2)

stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
df = pd.merge(stock_price, stock_price_forecast, on='ds', how='right')
fig1, ax1 = plt.subplots(figsize=(10, 6))
df.set_index('ds').plot(figsize=(16,8), color=['royalblue', "#34495e", "#e74c3c", "#e74c3c"], grid=True, ax=ax1)
st.pyplot(fig1)


'''
While the 3 year forecast we created above is pretty cool we don't want to make any trading decisions on it without backtesting the performance and a trading strategy.

In this section we will simulate as if Prophet existed back in 1980 and we used it to creat a monthly forecast through 2019. We will then use this data in the following section to simulate how various trading strategies did vs if we just bought and held on to the stock.
Before we simulate the monthly forecasts we need to add some columns to our stock_price dataframe we created in the beginning of this project to make it a bit easier to work with. We add month, year, month/year, and month/year_index.
'''
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

We will loop through unique month/year in the stock_price and fitting the Prophet model with the stock data available to that period and then forecasting out one month ahead. 
We continue to do this until we hit the last unique month/year. 
Finally we combine these forecasts into a single dataframe called stock_price_forecast. 
I save the results as it take a while to run and in case I need to reset I can pull the csv file instead of running the model again.
'''

if os.path.exists('./data/stock_price_forecast.csv'):
	stock_price_forecast = pd.read_csv('./data/stock_price_forecast.csv', parse_dates=['ds'])
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

    stock_price_forecast = reduce(lambda top, bottom: pd.concat([top, bottom], sort=False), forecast_frames)
    stock_price_forecast = stock_price_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    stock_price_forecast.to_csv('./data/stock_price_forecast.csv', index=False)

'''
Finally we combine our forecast with the actual prices and create a Percent Change column which will be used in our Trading Algorithms below. 
Lastly, I plot the forecasts with the actuals to see how well it did. As you can see there is a bit of a delay. 
It kind of behaves a lot like a moving average would.
'''

df = pd.merge(stock_price[['ds','y', 'month/year_index']], stock_price_forecast, on='ds')
df['Percent Change'] = df['y'].pct_change()
df.set_index('ds')[['y', 'yhat', 'yhat_lower', 'yhat_upper']].plot(figsize=(16,8), color=['royalblue', "#34495e", "#e74c3c", "#e74c3c"], grid=True)
st.pyplot(plt)


'''
We create four initial trading algorithms:
* **Hold**: Our bench mark.  This is a buy and hold strategy. Meaning we buy the stock and hold on to it until the end time period.
* **Prophet**: This strategy is to sell when our forecast indicates a down trend and buy back in when it iindicates an upward trend
* **Prophet Thresh**: This strategy is to only sell when the stock price fall below our yhat_lower boundary.
* **Seasonality**:  This strategy is to exit the market in August and re-enter in Ocober. This was based on the seasonality chart from above.
We simulating an initial investment of $1,000 dollars.  
'''
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

''' 
create an optimized threshold at each time point based on a rolling windows to incorporte changing thrsholds over time.
'''
if os.path.exists('./data/rolling_thresh.csv'):
    rolling_thresh = pd.read_csv('./data/rolling_thresh.csv', parse_dates=['ds'])
else:

    rolling_thresh = {}

    for num, index in enumerate(df['month/year_index'].unique()):
    
        rolling_performance = {}
        
        for roll in range(10, 400, 10):
                
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
        best_rolling_thresh = per_df.iloc[[-1]].max().idxmax()
        
        if num == len(df['month/year_index'].unique())-1:
            pass
        else:
            rolling_thresh[df['month/year_index'].unique()[num+1]] = best_rolling_thresh

    rolling_thresh = pd.DataFrame([rolling_thresh]).T.reset_index().rename(columns={'index':'month/year_index', 0:'Fcst Thresh'})
    rolling_thresh.to_csv('./data/rolling_thresh.csv', index=False)

rolling_thresh['Fcst Thresh'].plot(figsize=(16,8), grid=True)
df['yhat_optimized'] = pd.merge(df, rolling_thresh, 
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

