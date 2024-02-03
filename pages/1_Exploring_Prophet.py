import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from collections import Counter
import os
import toml

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

readme = dict(toml.load('./documents/readme.toml'))

st.title('Exploring Prophet')
'''
This app allows you to train, evaluate and optimize a Prophet model in just a few clicks.
All you have to do is to upload a time series dataset, and follow the guidelines in the sidebar to:
* __Prepare data__: Filter, aggregate, resample and/or clean your dataset.
* __Choose model parameters__: Default parameters are available but you can tune them.
Look at the tooltips to understand how each parameter is impacting forecasts.
* __Select evaluation method__: Define the evaluation process, the metrics and the granularity to
assess your model performance.
* __Make a forecast__: Make a forecast on future dates that are not included in your dataset,
with the model previously trained. \n
'''
st.header('1. Using yfinance to get stock data')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Stock parameter')
    with st.expander('Choose stock', expanded=True):
        # take user text input for stock or allow user to choose from a list of stocks
        input_stock = st.text_input('Please input Stock Ticker', '')
        stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
        selected_stock = st.selectbox('Or select dataset for prediction', stocks)

        if input_stock:
            selected_stock = input_stock
        else:
            selected_stock = selected_stock
        
    with st.expander('Choose date range', expanded=True):
        # START = "2015-01-01"
        START = st.date_input('Input Start Date', value=date(2015, 1, 1), min_value=date(2015, 1, 1), max_value=date(2022, 1, 1), key='start_date')
        TODAY = st.date_input('Input End Date', value=date.today(), min_value=date(2015, 1, 1), max_value=date.today(), key='end_date')
        START = START.strftime("%Y-%m-%d")
        TODAY = TODAY.strftime("%Y-%m-%d")

    ## cache the data
    @st.cache_data(ttl=3600, show_spinner="........")
    def load_data(ticker, START, TODAY):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    button = st.button('Load Data')
with col2:
    if button:
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock, START, TODAY)
        data_load_state.text('Loading data... done!')

        st.subheader(f'{selected_stock} Data')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(xaxis_rangeslider_visible=True)
        # col2.write(f'{selected_stock} Stock Prices')
        st.plotly_chart(fig)



st.header('2. Using Prophet to model stock prices into the future')
'''
We will explore modeling with [prophet](https://facebook.github.io/prophet/), a library for demand forecasting, and use Google stock data from 2013-01-01 to 2023-12-31   
'''
if os.path.exists('./data/GOOG.csv'):
    data = pd.read_csv(r'./data/GOOG.csv', parse_dates=['Date'])
else:
# Read the training file into a DataFrame
    data = load_data('GOOG', '2013-01-01', '2023-12-31')
    data.to_csv(r'./data/GOOG.csv', index=False)  
    
    

st.subheader('Tunable Parameters')
'''
Prophet uses a decomposable time series model with three main model components: growth, seasonality and holidays.   
            They are combined using the equation  
            $$y(t) = g(t) + s(t) + h(t) + e(t)$$  
            - g(t) represents the growth function which models non-periodic changes,  
            - s(t) represents periodic changes due to weekly or yearly seasonality,  
            - h(t) represents the effects of holidays  
            - e(t) represents the error term
'''

col3, col4 = st.columns(2)
with col3:
    with st.expander('Dates', expanded=False):
        START = st.date_input('Input Start Date', value=date(2015, 1, 1), min_value=date(2015, 1, 1), max_value=date(2022, 1, 1), key='start')
        TODAY = st.date_input('Input End Date', value=date.today(), min_value=date(2015, 1, 1), max_value=date.today(), key='end')
        START = START.strftime("%Y-%m-%d")
        TODAY = TODAY.strftime("%Y-%m-%d")
        years = st.slider('Year into future', 1, 5, 3, 1, key='years')
    with st.expander('Seasonality', expanded=False):
        seasonality_mode = st.selectbox('Seasonality Mode', ('additive', 'multiplicative'), key='seasonality_mode', help=readme["tooltips"]["seasonality"])
        daily_seasonality = st.checkbox('Daily Seasonality', value=False, key='daily_seasonality')
        weekly_seasonality = st.checkbox('Weekly Seasonality', value=True, key='weekly_seasonality')
        yearly_seasonality = st.checkbox('Yearly Seasonality', value=True, key='yearly_seasonality')
    with st.expander('Holidays', expanded=False):
        holidays = st.checkbox('Holidays', value=True, key='holidays')
        if holidays:
            country = st.selectbox('Country', ('US', 'UK', 'CA'), key='country')
    with st.expander('Other', expanded=False):
        interval_width = st.slider('Interval Width', 0.0, 1.0, 0.95, 0.05, key='interval_width')
        changepoint_range = st.slider(
            "changepoint_range",
            value=0.8,
            max_value=1.0,
            min_value=0.0,
            step = 0.05,
            format="%.2f",
            help=readme["tooltips"]["changepoint_range"],
        )
        growth = st.selectbox('Growth', ('linear', 'logistic', 'flat'), key='growth',
            help=readme["tooltips"]["growth"],)


    # Predict forecast with Prophet.
    stock_price = data[['Date','Close']]
    stock_price = stock_price.rename(columns={"Date": "ds", "Close": "y"})

    button2 = st.button('Train Model!')
    
if button2:
    training_state = st.text('Training model...')
    # get data in date range
    stock_price = stock_price[(stock_price['ds'] >= START) & (stock_price['ds'] <= TODAY)]

    # instantiate the model and set parameters
    m = Prophet(
        interval_width=interval_width,
        growth=growth,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode
    )
    if holidays:
        m.add_country_holidays(country_name=country)
    m.fit(stock_price)
    training_state.text('Training model... done!')

    # helper method to make dataframe with future dates
    future = m.make_future_dataframe(periods=years * 365, 
        freq='d', 
        include_history=True)


    # Show and plot forecast
    predicting_state = st.text(f'Predicting {years} years of data...')

    @st.cache_resource(ttl=3600, show_spinner="Training and Fitting ........")
    def train_model(_m, future):
        future_boolean = future['ds'].map(lambda x : True if x.weekday() in range(0, 5) else False)
        future = future[future_boolean] 
        forecast = m.predict(future)
        return forecast

    forecast = train_model(m, future)
    predicting_state.text(f'Predicting {years} years of data... done!')

    col5, col6 = st.columns(2)
    with col5:
        st.subheader(f'Forecast')
        '''The Prophet plot functions creates a graph of our actuals and forecast. Black dots are actual values'''
        fig1 = plot_plotly(m, forecast, xlabel='Date', ylabel='Price')
        st.plotly_chart(fig1)
        
    with col6:
        st.subheader("Forecast components")
        '''The Prophet plot_components provides a breakdown of the components. '''
        fig2 = m.plot_components(forecast, figsize=(16, 9))
        st.write(fig2)

