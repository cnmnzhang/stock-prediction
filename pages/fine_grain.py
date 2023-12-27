import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from collections import Counter
import os
st.markdown("# Fine grain forecasting 🎈")

'''
generate a large number of fine-grained forecasts at the store-item level in an efficient manner 
leveraging the distributed computational power of Databricks.  
Exploring [prophet](https://facebook.github.io/prophet/), a library for demand forecasting.  

Prophet uses a decomposable time series model with three main model components: growth, seasonality and holidays. 
            They are combined using the equation
            $$y(t) = g(t) + s(t) + h(t) + e(t)$$
            where g(t) represents the growth function which models non-periodic changes, s(t) represents periodic changes due to weekly or yearly seasonality, h(t) represents the effects of holidays, and e(t) represents the error term


training dataset: 5-years of store-item unit sales data for 50 items across 10 different stores.  
[Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview)

'''

# Read the training file into a DataFrame
train = pd.read_csv(r'./data/train.csv', parse_dates=['date'])


st.subheader("Exploratory Analysis")
st.markdown('''Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed 
            with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1), 
            and then a steady pickup over the week heading back to the Sunday high. 
            This pattern seems to be pretty stable across the five years of observations:''')

raw, yearly_tab, monthly_tab, weekday_tab = st.tabs(["Raw", "Yearly", "Monthly", "Weekday"])
raw1, raw2 = raw.columns(2)
raw1.write('Data')
raw1.write(f'shape: {train.shape}')
raw1.write(train.head())
# plot item sales bargraph across all stores
raw1.write('Item Sales Across All Stores')
raw1.bar_chart(train.groupby('item')['sales'].sum().reset_index().sort_values(by='sales', ascending=False).set_index('item').sort_values(by='sales', ascending=False))

# create dataframe for top 5 most popular item at each store, where each row is a store and we have a column for top 1, top 2, top 3, top 4, and top 5. the value in top1 is the item number of the most popular item
top5 = train.groupby(['store', 'item'])['sales'].mean().reset_index().sort_values(by='sales', ascending=False).groupby('store').head(5).sort_values(by=['store', 'sales'], ascending=[True, False]).reset_index(drop=True)
top5['rank'] = top5.groupby('store').cumcount() + 1
top5 = top5.pivot(index='store', columns='rank', values='item').reset_index()
top5.columns = ['store', 'top1', 'top2', 'top3', 'top4', 'top5']
raw2.write('Top 5 most popular items at each store')
raw2.write(top5)


# View Yearly Trends
yearly_tab.markdown('''The yearly trends show a steady increase in total sales over the five years of data.''')
train['year'] = train['date'].dt.year
yearly_trends = train.groupby(['year'])['sales'].sum().rename('total sales').reset_index()
yearly_tab.line_chart(yearly_trends[['year', 'total sales']].set_index('year'))

# View Monthly Trends
monthly_tab.markdown('''The monthly trends show a clear seasonal pattern with a peak in the summer months and a trough in the winter months.''')
train['month'] = train['date'].dt.month
monthly_trends = train.groupby(['month'])['sales'].mean().rename('average monthly sales').reset_index()
monthly_tab.line_chart(monthly_trends[['month', 'average monthly sales']].set_index('month'))

# View Weekday Trends
weekday_tab.markdown('''The weekday trends show a clear weekly pattern with a peak on Sunday and a trough on Monday.''')
train['weekday'] = train['date'].dt.dayofweek
weekday_trends = train.groupby(['weekday'])['sales'].mean().rename('average weekly sales').reset_index()
weekday_trends['weekday'] = weekday_trends['weekday'].replace({'0': '0-Mon', '1': '1-Tue', '2': '2-Wed', '3': '3-Thu', '4': '4-Fri', '5': '5-Sat', '6': '6-Sun'}) 
weekday_tab.line_chart(weekday_trends[['weekday', 'average weekly sales']].set_index('weekday'))

# View Store Trends
st.markdown('# Train for all item-store combinations')

train['ds'] = train['date']
train['y'] = train['sales']


# Forecast for All Store-Item Combinations
@st.cache_data(ttl=3600, show_spinner="........")
def forecast_store_item(history):
    history = history.dropna()
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(history)

    future = model.make_future_dataframe(periods=90, freq='D', include_history=True)
    forecast = model.predict(future)

    f_pd = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
    h_pd = history[['ds', 'store', 'item', 'y']].set_index('ds')
    results_pd = f_pd.join(h_pd, how='left')
    results_pd.reset_index(level=0, inplace=True)

    results_pd['store'] = history['store'].iloc[0]
    results_pd['item'] = history['item'].iloc[0]

    return results_pd[['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

# Apply Forecast Function to Each Store-Item Combination
# results = train.groupby(['store', 'item']).apply(forecast_store_item).reset_index(drop=True)


# # Persist Forecast Output
# results.to_csv('forecasts.csv', index=False)


# st.write(results.head())
