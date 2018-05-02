#基本的なライブラリの読み込み
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#統計
import statsmodels.api as sm
from scipy import stats

#データの読み込み
data = pd.read_csv('arma.csv')
ts = data['y1']

#y1の標本自己相関と標本偏自己相関をプロット
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(221)
sm.graphics.tsa.plot_acf(ts, lags=20, ax=ax1)
ax2 = fig.add_subplot(222)
sm.graphics.tsa.plot_pacf(ts, lags=20, ax=ax2)

plt.show()

#y1に対する各モデルのフィット
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
ar = AR(ts)
arma_11 = ARMA(ts, order=(1,1))
arma_12 = ARMA(ts, order=(1,2))
arma_21 = ARMA(ts, order=(2,1))
arma_22 = ARMA(ts, order=(2,2))
