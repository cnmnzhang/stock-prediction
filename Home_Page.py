import streamlit as st

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
from prophet.plot import plot_plotly

# from pages.helper.utility import add_logo

add_logo()

st.title('Portfolio')
st.write('''Just quick explorations of topics that pique my interest''')

with st.expander('December: Exploring streamlit', expanded=False):
   st.write('''
            I am discovering stream lit and playing around with the possibilities! 
            ''')

with st.expander('January: Exploring Prophet', expanded=False):
   st.write('''
            Inspired by my brother when we were home for the holidays.
            He introduced me to bid books, where a level 2 bid book contains trading bid and ask prices and volumes across small time intervals.
            
            This data provides insights into the market's direction and we can use the data in the book to derive features 
            to predict the direction of the market in the next 30 minutes!
            
            For example, buy volumn is the total number of shares that are bought at the ask price and sell volume is the total number of shares that are sold at the bid price.
            We can take the difference between the two divided by the total volume to get a ratio which is **nonlinear**
            
            A value closer to 1 means the market is more likely to go up and a value closer to -1 means the market is more likely to go down.
            We can train models like NN or LR on this feature to predict market direction and help us decide when to enter the market. 
            Since bid book data is relatively granular, we can run the trained model every 5 minutes to help us make a decision in that moment.
            Models trained on snapshot data and not timeseries are particularly useful for short term action!''')
   
   st.markdown('### 1. Using Prophet for Forecasting')
   st.write(''' 
            I explored yfinance api for retrieving stock data and facebook prophet for forecasting with prebuilt models!''')
            
               

   st.markdown('### 2: Using Prophet for Trading')
   st.write('''I want to learn when to buy and sell! I explored different techniques: 
            - Hold: Buy and don't let go! 
            - Prophet: Sell when our forecast indicates a down trend and buy back when it indicates an upward trend 
            - Threshold: Sell when the stock price falls below our lower forecast boundary. 
            - Seasonality: This strategy is to exit the market in August and re-enter in Ocober. This was based on the seasonality chart from above. 
            '''
            )

   st.markdown('### 3. Using XGBoost for Prediction')
   st.write('''
            Prophet was really cool to learn about, but I wanted to see how models I knew of (XGBoost) would perform when it comes to prediction. 
            '''
            )

with st.expander('February', expanded=False):
   st.markdown('### 4: Bifurcation Diagram of a Chaotic Dynamical System')
   st.write('''
            My friend mentioned chaos theory casually, and I wanted in.  
            '''
            )

         



            