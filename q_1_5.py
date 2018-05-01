# 基本のライブラリを読み込む
import numpy as np
import pandas as pd
from scipy import stats

# グラフ描画
from matplotlib import pylab as plt
import seaborn as sns

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 15, 6

#統計モデル
import statsmodels.api as sm

#日付形式でデータを読み込む
data = pd.read_csv('data_original.csv', index_col=[0], parse_dates=[0])
ts = data['indprod']

"""
# topixをプロットする
plt.plot(ts)
plt.show()
"""

#　対数差分の計算
logtest = np.log(ts) - np.log(ts.shift())
logdiff = logtest.dropna()

# 自己相関の計算、プロット
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(logdiff, lags=20, ax=ax1)

#　偏自己相関の計算
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(logdiff, lags=20, ax=ax2)

# Ljung-Box検定
ljungbox_test = sm.stats.diagnostic.acorr_ljungbox(logdiff, 10)
ljungbox = pd.DataFrame({'Q(m)': ljungbox_test[0],
'P-value': ljungbox_test[1]})
print(ljungbox)
