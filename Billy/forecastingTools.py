import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def ts_display(ts, lags: int=25):
    """Plot timeseries plot, ACF and PACF plots 

    Args:
        ts (pd.core.frame.DataFrame | pd.core.series.Series): time series to be plotted
        lags (int, optional): Lags to be calculated in ACF y PACF. Defaults to 25:int.
    """
    grid = plt.GridSpec(2, 2)
    # Plot series as is
    plt.subplot(grid[0, :])
    plt.tight_layout(pad=4.0)
    plt.plot(ts)
    # Plot ACF
    ax = plt.subplot(grid[1, 0])
    sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    # Plot PACF
    ax = plt.subplot(grid[1, 1])
    sm.graphics.tsa.plot_pacf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')