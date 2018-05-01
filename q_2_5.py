# 基本のライブラリを読み込む
import numpy as np
import pandas as pd
from scipy import stats

# グラフ描画
from matplotlib import pylab as plt

#統計モデル
import statsmodels.api as sm

#日付形式でデータを読み込む
data = pd.read_csv('data_original.csv', index_col=[0], parse_dates=[0])
ts = data['indprod']

#　対数差分の計算
logtest = np.log(ts) - np.log(ts.shift())
logdiff = logtest.dropna()

#自己相関と偏自己相関
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(221)
sm.graphics.tsa.plot_acf(logdiff, lags=20, ax=ax1)
ax2 = fig.add_subplot(222)
sm.graphics.tsa.plot_pacf(logdiff, lags=20, ax=ax2)


# 差分系列のAR(4)によるフィット、誤差の表示
from statsmodels.tsa.ar_model import AR
AR_4 = AR(logdiff).fit(maxlag=4)
ax3 = fig.add_subplot(223)
sm.graphics.tsa.plot_acf(AR_4.resid, lags=20, ax=ax3)

# 差分系列のARMA(1,2)によるフィット、誤差の表示
from statsmodels.tsa.arima_model import ARMA
ARMA_1_2 = ARMA(logdiff, order=(1,2)).fit(dist=False)
ax4 = fig.add_subplot(224)
sm.graphics.tsa.plot_acf(ARMA_1_2.resid, lags=20, ax=ax4)


# 残差のかばん検定
from statsmodels.stats.diagnostic import acorr_ljungbox as ljbox
ar_boxtest =  ljbox(AR_4.resid, lags=10)
arma_boxtest = ljbox(ARMA_1_2.resid, lags=10)

# 対応するp値の計算
pvalue_ar = stats.chisqprob(ar_boxtest[0][9], 6)
pvalue_arma = stats.chisqprob(arma_boxtest[0][9], 7)

print(pvalue_ar)
print(pvalue_arma)
