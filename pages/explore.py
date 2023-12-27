
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


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
period = st.sidebar.slider('Days of prediction:', 1, 365)

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


col1, col2 = st.columns(2)

with col1:
	st.subheader('Raw data')
	st.write(data.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()

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
future = m.make_future_dataframe(periods=period, 
    freq='d', 
    include_history=False)

future_boolean = future['ds'].map(lambda x : True if x.weekday() in range(0, 5) else False)
future = future[future_boolean] 

forecast = m.predict(future)

with col2:
	# Show and plot forecast
	st.subheader(f'Forecast {period} days of data')
	st.write(forecast.tail())
	'''The plot functions creates a graph of our actuals and forecast'''
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)

	st.write("Forecast components")
	'''plot_components provides us a graph of our trend and seasonality.'''
	fig2 = m.plot_components(forecast)
	st.write(fig2)