import streamlit as st

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
from prophet.plot import plot_plotly

with st.expander('December', expanded=False):
   st.header('0: Exploring streamlit')
   st.write('''
            I am discovering stream lit and playing around with the possibilities! 
            ''')

with st.expander('January', expanded=False):
   st.header('1: Exploring Prophet')
   st.write('''
            I want to learn about applications of modeling, since my brother was working on a personal project while we were both home. 
            He was working with level 2 bid book trading data which contains bid and ask prices and volumes across small time intervals.
            His goal was to predict the direction of the market in the next 30 minutes.
            he uses the buy volume - sell volume divides by the total volume to get a ratio which is nonlinear
            a value closer to 1 means the market is more likely to go up and a value closer to -1 means the market is more likely to go down.
            He evaluated neural networks and linear regressions for predict if we should enter the market. 
            This is evaluated every 5 minutes and we exit the market 30 minutes later because the model contains this level of knowledge
            
            His model is useful for short term since the data is snapshot based and not time series based.
            
            This week I learned to: 
            - yfinance api to get stock data 
            - explored facebook prophet 
               ''')

   st.header('2: Using Prophet for Trading')
   st.write('''I want to learn when to buy and sell! I explored different techniques: 
            
            - Hold: Buy and don't let go! 
            - Prophet: Sell when our forecast indicates a down trend and buy back when it indicates an upward trend 
            - Threshold: Sell when the stock price falls below our lower forecast boundary. 
            - Seasonality: This strategy is to exit the market in August and re-enter in Ocober. This was based on the seasonality chart from above. 
            '''
            )

   st.header('3: Using XGBoost for Prediction')
   st.write('''
            Prophet was really cool to learn about, but I wanted to see how models I knew of (XGBoost) would perform when it comes to prediction. 
            '''
            )

with st.expander('February', expanded=False):
   st.header('4: Bifurcation Diagram of a Chaotic Dynamical System')
   st.write('''
            My friend told me about chaos theory! I wanted to learn more about it. 
            '''
            )

         



            