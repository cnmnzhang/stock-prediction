import numpy as np
import plotly.graph_objects as go
from scipy.special import erfc
import yfinance as yf
import streamlit as st

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://assets.stickpng.com/thumbs/5a5a8b8614d8c4188e0b08e6.png);
                background-repeat: no-repeat;
                background-size: 80%;
                height: auto;
                padding-bottom: 120px;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Projects";
                margin-left: 20px;
                margin-top: 20px;
                margin-bottom: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

## cache the data
@st.cache_data(ttl=3600, show_spinner="........")
def load_data(ticker, START, TODAY):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

z = np.linspace(-8,8,200)

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

def logistic(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

selu_alpha = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
selu_scale = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e**2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)

def selu(z, scale=selu_scale, alpha=selu_alpha):
    return scale * elu(z, alpha)

def plot_function(func, title, alpha=None):
    fig = go.Figure()
    if alpha:
        fig.add_trace(go.Scatter(x=z, y=func(z, alpha=alpha), mode='lines', line=dict(color='red', width=3)))
    else:
        fig.add_trace(go.Scatter(x=z, y=func(z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z',width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')

    return fig

def plot_function_derivative(func, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=derivative(func, z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z', width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    return fig