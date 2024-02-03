import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st
import yfinance as yf
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import toml

# save xgb model
import joblib

# Time series decomposition
# !pip install stldecompose
# from stldecompose import decompose
# https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction

# Chart drawing

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
readme = dict(toml.load('./documents/readme.toml'))


# Change default background color for all visualizations
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'


fig, ax = plt.subplots(figsize=(16,8))
## cache the data
@st.cache_data(ttl=3600, show_spinner="........")
def load_data(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.title('Using XGBoost for Prediction')
st.write('Training XGBRegressor to predict future prices of stocks using technical indicator as features.')


st.sidebar.title("Data")

# Load data
with st.sidebar.expander("Dataset", expanded=False):
    # st.button('Refresh Data', help=readme["tooltips"]["refresh_data"])
    dataset_name = st.selectbox(
            "Select a toy dataset",
            ["GOOG", "AAPL", "MSFT", "GME"],
    )
if os.path.exists('./data/train-'+ dataset_name + '.csv'):
    data = pd.read_csv(r'./data/GOOG.csv', parse_dates=['Date'])
else:
# Read the training file into a DataFrame
    data = load_data(dataset_name, '2013-01-01', '2023-12-31')
    data.to_csv(r'./data/train-'+ dataset_name + '.csv', index=False)  
    
    

st.sidebar.title("Data Preparation")
# select option for data date granularity of monthly, weekly, daily, 
with st.sidebar.expander("Date Granularity", expanded=False):
    granularity = st.sidebar.selectbox('Select data granularity', ('Daily', 'Weekly', 'Monthly'), key='granularity')
    # Convert 'Date' column to categorical based on selected granularity
    if granularity == 'Monthly':
        data['DateCategory'] = data['Date'].dt.to_period('M').astype('category')
    elif granularity == 'Weekly':
        data['DateCategory'] = data['Date'].dt.to_period('W').astype('category')
    else:  # Daily
        data['DateCategory'] = data['Date'].dt.date.astype('category')

st.header('2. Feature Engineering')
with st.expander('Moving Averages', expanded=False):
    st.markdown(
    """
    I'm calculating few moving averages to be used as features: $SMA_{5}$, $SMA_{10}$, $SMA_{15}$, $SMA_{30}$ and $EMA_{9}$.  
    [Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp) (MA) help to smooth out stock prices on a chart by filtering out short-term price fluctuations. We calculate moving averages over a defined period of time e.g. last 9, 50 or 200 days. There are two (most common) averages used in technical analysis which are:
        * *Simple Moving Average (SMA)* - a simple average calculated over last N days e.g. 50, 100 or 200,
        * *Exponential Moving Average (EMA)* - an average where greater weights are applied to recent prices.
    """
    )
data['EMA_9'] = data['Close'].ewm(9).mean().shift()
data['SMA_5'] = data['Close'].rolling(5).mean().shift()
data['SMA_10'] = data['Close'].rolling(10).mean().shift()
data['SMA_15'] = data['Close'].rolling(15).mean().shift()
data['SMA_30'] = data['Close'].rolling(30).mean().shift()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.Date, y=data.EMA_9, name='EMA 9'))
fig.add_trace(go.Scatter(x=data.Date, y=data.SMA_5, name='SMA 5'))
fig.add_trace(go.Scatter(x=data.Date, y=data.SMA_10, name='SMA 10'))
fig.add_trace(go.Scatter(x=data.Date, y=data.SMA_15, name='SMA 15'))
fig.add_trace(go.Scatter(x=data.Date, y=data.SMA_30, name='SMA 30'))
fig.add_trace(go.Scatter(x=data.Date, y=data.Close, name='Close', opacity=0.2))
st.plotly_chart(fig)

# st.subheader('Bollinger Bands')
# st.write("I'm calculating Bollinger Bands to be used as features: $BB_{up}$, $BB_{down}$ and $BB_{width}$.")
# with st.expander('More info on Bollinger Bands', expanded=False):
#     st.markdown(
#     """
#     [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp) (BB) are a type of price envelope developed by John Bollinger. It consists of three lines:
# * *Upper band* - SMA plus 2 standard deviations,
# * *Lower band* - SMA minus 2 standard deviations,
# * *Middle band* - 20-day simple moving average (SMA).
#     """
#     )
# data['BB_up'] = data['SMA_15'] + 2 * data['Close'].rolling(20).std()
# data['BB_down'] = data['SMA_15'] - 2 * data['Close'].rolling(20).std()
# data['BB_width'] = data['BB_up'] - data['BB_down']

with st.expander('Relative Strength Index', expanded=False):
    st.markdown(
    """
    I'll add [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (RSI) which indicates magnitude of recent price changes. 
    Typically RSI value of 70 and above signal that a stock is becoming overbought/overvalued, meanwhile value of 30 and less can mean that it is oversold. Full range of RSI is from 0 to 100."""
    )

def relative_strength_idx(data, n=14):
    close = data['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

data['RSI'] = relative_strength_idx(data).fillna(0)

fig = go.Figure(go.Scatter(x=data.Date, y=data.RSI, name='RSI'))
st.plotly_chart(fig)

with st.expander('oving Average Convergence Divergence', expanded=False):
    st.markdown(
    """
    I'll add MACD indicator which serves as a signal to buy or sell.
    [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp) (MACD) is an indicator which shows the relationship between two exponential moving averages i.e. 12-day and 26-day EMAs. We obtain MACD by substracting 26-day EMA (also called *slow EMA*) from the 12-day EMA (or *fast EMA*).
    You can more more about entry/exit signals that can be read from MACD under [this link](https://www.investopedia.com/terms/m/macd.asp).
    """
    )

EMA_12 = pd.Series(data['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(data['Close'].ewm(span=26, min_periods=26).mean())
data['MACD'] = pd.Series(EMA_12 - EMA_26)
data['MACD_signal'] = pd.Series(data.MACD.ewm(span=9, min_periods=9).mean())

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=data.Date, y=data.Close, name='Close'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
fig.add_trace(go.Scatter(x=data.Date, y=data['MACD'], name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=data.Date, y=data['MACD_signal'], name='Signal line'), row=2, col=1)
st.plotly_chart(fig)

st.subheader('Data Preparation Last Steps')

'''
### Shift label column
Because I want to predict the next day price, after calculating all features for day $D_{i}$, I shift Close price column by -1 rows. 
After doing that, for day $D_{i}$ we have features from the same timestamp e.g. $RSI_{i}$, but the price $C_{i+1}$ from upcoming day.

### Drop invalid samples
Because of calculating moving averages and shifting label column, few rows will have invalid values i.e. we haven't calculated $SMA_{10}$ for the first 10 days. 
Moreover, after shifting Close price column, last row price is equal to 0 which is not true. 
I remove these samples

### Data Split
I split stock data frame into three subsets. Default values: training ($70\%$), validation ($15\%$) and test ($15\%$) sets. 
All three frames have been ploted in the chart below.
'''
data['Close'] = data['Close'].shift(-1)
data = data.iloc[33:] # Because of moving averages and MACD line
data = data[:-1]      # Because of shifting close price
data.index = range(len(data))

with st.sidebar.expander("Train/Valid/Test Split", expanded=True):
    test_size = st.sidebar.slider('Test size', 0.05, 0.4, 0.15, 0.05)
    valid_size = st.sidebar.slider('Validation size', 0.05, 0.4, 0.15, 0.05)


test_split_idx  = int(data.shape[0] * (1-test_size))
valid_split_idx = int(data.shape[0] * (1-(valid_size+test_size)))

train_data  = data.loc[:valid_split_idx].copy()
valid_data  = data.loc[valid_split_idx+1:test_split_idx].copy()
test_data   = data.loc[test_split_idx+1:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data.Date, y=train_data.Close, name='Training'))
fig.add_trace(go.Scatter(x=valid_data.Date, y=valid_data.Close, name='Validation'))
fig.add_trace(go.Scatter(x=test_data.Date,  y=test_data.Close,  name='Test'))
st.plotly_chart(fig)

st.header('3. Train')

features = ['EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']


# define data
y_train = train_data['Close'].copy()
X_train = train_data[features].copy()

y_valid = valid_data['Close'].copy()
X_valid = valid_data[features].copy()

y_test  = test_data['Close'].copy()
X_test  = test_data[features].copy()

# side bar multiselect for parameters with expander for each key
st.sidebar.title('Finetuning')
parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
    }
with st.sidebar.expander("Grid Search Parameters", expanded=False):
    for key in parameters.keys():
            parameters[key] = st.sidebar.multiselect(key, parameters[key], default=parameters[key])
parameters['random_state'] = [st.sidebar.slider('random_state', 0, 100, 42, 1)]

eval_set = [(X_train, y_train), (X_valid, y_valid)]

model_path = "./models/xgboost_model.joblib"
model_exists = os.path.exists(model_path)
retrain = st.button('Retrain Model', help=readme["tooltips"]["launch_forecast"])
if model_exists and not retrain:
    model = joblib.load(model_path)
else:
    training_status = st.text('Training model...')


    model = xgb.XGBRegressor(objective='reg:squarederror', enable_categorical=True)
    clf = GridSearchCV(model, parameters)
    # st.write(X_train.dtypes)
    clf.fit(X_train, y_train, verbose=False, eval_set=eval_set)
    
    training_status.test(f'Training model...Done !!!')
    st.write(f'Best params: {clf.best_params_}')
    st.write(f'Best validation score = {clf.best_score_}')

    model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror', enable_categorical=True)
    retraining_status = st.text(f'Training model with best params...')
    model.fit(X_train, y_train, verbose=False, eval_set=eval_set)
    retraining_status.text(f'Training model with best params...Done!!!')
    
    joblib.dump(model, 'xgboost_model.joblib')
    
st.header('4. Model Feature Importance')
plot_importance(model)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
# fig = plot_importance(model)
# st.plotly_chart(fig)

st.header('5. Test and Evaluate')
y_pred = model.predict(X_test)


with st.expander("More info on Metrics", expanded=False):
    st.markdown(
    """
    The following metrics can be computed to evaluate model performance:
    * __Mean Squared Error (MSE)__: Measures the average squared difference between forecasts and true values.
    This metric is not ideal with noisy data,
    because a very bad forecast can increase the global error signficantly as all errors are squared.
    * __Root Mean Squared Error (RMSE)__: Square root of the MSE.
    MSE measures the average squared difference between forecasts and true values, which is not ideal for noisy data
    RMSE is more robust to outliers than the MSE, as the square root limits the impact of large errors in the global error.
    * __Mean Absolute Error (MAE)__: Measures the average absolute error.
    This metric can be interpreted as the absolute average distance between the best possible fit and the forecast.
    * __Mean Absolute Percentage Error (MAPE)__: Measures the average absolute size of each error in percentage
    of the truth. This metric is not ideal for low-volume forecasts,
    because being off by a few units can increase the percentage error signficantly.
    It can't be calculated if the true value is 0 (here samples are excluded from calculation if true value is 0).
    * __R2 (R2)__: Measure of fit, not how accurate the predictions are
    Proportion of the variance in the dependent variable that is predictable from the independent variable
    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)
    """
    )
col1, col2, col3, col4 = st.columns(4)
col1.subheader('**RMSE:**')
col1.write(f'{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
col2.subheader('**MAE:**')
col2.write(f'{mean_absolute_error(y_test, y_pred):.2f}')
col3.subheader('**MAPE:**')
col3.write(f'{mean_absolute_percentage_error(y_test, y_pred):.2f}')
col4.subheader('**R2:**')
col4.write(f'{r2_score(y_test, y_pred):.2f}')


    
# metrics = st.sidebar.multiselect('Metrics', ['mean_absolute_percentage_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score'], default=['mean_squared_error', 'mean_absolute_error', 'r2_score'])
# if 'mean_squared_error' in metrics:
#     st.write(f'RMSE = {mean_squared_error(y_test, y_pred)}')
# if 'mean_absolute_error' in metrics:
#     st.write(f'MAE = {mean_absolute_error(y_test, y_pred)}')
# if 'mean_absolute_percentage_error' in metrics:
#     st.write(f'MAPE = {mean_absolute_percentage_error(y_test, y_pred)}')
# if 'r2_score' in metrics:
#     st.write(f'R2 = {r2_score(y_test, y_pred)}')

predicted_prices = data.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=data.Date, y=data.Close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=predicted_prices.Close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.Date,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

st.plotly_chart(fig)





# %%time

# model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
# model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# plot_importance(model)