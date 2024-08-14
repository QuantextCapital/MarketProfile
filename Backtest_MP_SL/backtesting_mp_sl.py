import pandas as pd
from backtesting import Strategy, Backtest
from tpo_helper_simple import get_ticksize, get_mean, get_rf, get_context, get_dayrank
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
from def_btextra import btextra, equity_curve_plot, backtest_plot

df = pd.read_pickle(r'C:\Users\alex1\OneDrive\Public\NIFTY50-Minute.pkl')
split = int(len(df) / 2)
df_insample = df.head(split)
df_outsample = df.tail(split)

mode = 'tpo'
freq = 15
avglen = 30

ticksz = get_ticksize(df_outsample, freq=freq)
mean_val = get_mean(df_outsample, avglen=avglen, freq=freq)
trading_hr = mean_val['session_hr']

df_outsample = get_rf(df_outsample.copy())
df_outsample['datetime'] = df_outsample.index

dfresample = df_outsample.resample(str(freq) + 'min').agg({'datetime': 'last', 'Open': 'first',
                                                           'High': 'max',
                                                           'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
dfresample = dfresample.dropna()

dfcontext = get_context(dfresample, freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr)
dfmp_list = dfcontext[0]
df_distribution = dfcontext[1]
df_ranking = get_dayrank(df_distribution.copy(), mean_val)
ranking = df_ranking[0]

dfresample['dateonly'] = dfresample['datetime'].dt.date
ranking['dateonly'] = ranking['date'].dt.date
merged_df = pd.merge(dfresample, ranking, on='dateonly', how='left')

ranking2 = ranking.copy()

ranking2['dateonly1'] = ranking2['date'].dt.date
ranking2['dateonly'] = ranking2['dateonly'].shift(-1)

merged_df2 = pd.merge(dfresample, ranking2, on='dateonly', how='left')
merged_df2 = merged_df2[['datetime', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Volume_x', 'rf',
                         'tpocount', 'VAH', 'POC', 'VAL', 'ranged', 'Single_Prints', 'daytype_num', 'power1']]
merged_df2.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'rf', 'tpocount',
                      'VAH', 'POC', 'VAL', 'ranged', 'Single_Prints', 'daytype_num', 'power1']
merged_df2.set_index('datetime', inplace=True)
merged_df2 = merged_df2.dropna()

merged_df2['date'] = merged_df2.index
merged_df2['time'] = merged_df2['date'].dt.time
merged_df2['time'] = merged_df2['time'].astype(str)
merged_df2['time'] = merged_df2['time'].str.replace(':', '')
merged_df2['time'] = merged_df2['time'].astype(int)

# Trading logic to create signals inside pandas dataframe

merged_df2['long'] = np.where((merged_df2['Close'] > merged_df2['VAL']) &
                              (merged_df2['Close'].shift(1) < merged_df2['VAL']) &
                              # (merged_df2['Close'].shift(2) < merged_df2['VAL']) &
                              (merged_df2['power1'] > 0) & (merged_df2['power1'] < 100) &
                              (merged_df2['time'] >= 95900) & (merged_df2['time'] <= 142900)
                              ,
                              1, 0)

merged_df2['short'] = np.where((merged_df2['Close'] < merged_df2['VAH']) &
                               (merged_df2['Close'].shift(1) > merged_df2['VAH']) &
                               (merged_df2['power1'] < -1) & (merged_df2['power1'] > -100) &
                               # (merged_df2['Close'].shift(2) > merged_df2['VAH']) &
                               (merged_df2['time'] >= 95900) & (merged_df2['time'] <= 142900)
                               ,
                               -1, 0)


class EightyPercent(Strategy):
    def init(self):
        pass

    def next(self):
        # Here bulk of the strategy is written
        if not self.position:
            if self.data.long[-1] == 1:
                self.buy()
                self.entry_price_buy = self.data.Close[-1]
            elif self.data.short[-1] == -1:
                self.sell()
                self.entry_price_sell = self.data.Close[-1]
        else:
            if self.position.is_long:
                if self.data.time == 152900 or (self.data.Close[-1] > self.data.VAH[-1] and
                                                self.data.Close[-2] < self.data.VAH[-2]
                    or(self.data.Low[-1] < self.entry_price_buy - (self.entry_price_buy*0.0025))
                    or(self.data.Close[-1] > self.entry_price_buy + (self.entry_price_buy* 0.01))
                ):
                    self.position.close()
            if self.position.is_short:
                if self.data.time == 152900 or (self.data.Close[-1] < self.data.VAL[-1] and
                                                self.data.Close[-2] > self.data.VAL[-2]
                    or(self.data.High[-1] > self.entry_price_sell + (self.entry_price_sell* 0.0025))
                    or(self.data.Close[-1] < self.entry_price_sell - (self.entry_price_sell*0.01))

                ):
                    self.position.close()


backtest = Backtest(merged_df2, EightyPercent, cash=50_000, trade_on_close=True)
stats = backtest.run()
print(stats)

# part 2 visualisation
trades = stats['_trades']
btreport = btextra(merged_df2, trades.copy(), freq)
print(btreport[0])


with open('backtestreport_v2.txt', 'w') as f:
    f.write(str(stats)+'\n'+btreport[0]+'Note: Stop Loss = 0.4%')

fig = btreport[1]
plot(fig)

ec = stats['_equity_curve']
eqfig = equity_curve_plot(ec)
plot(eqfig)
merged_df2.rename(columns={'power1':'indicator'},inplace=True)
backtest_fig = backtest_plot(merged_df2,trades.copy())
plot(backtest_fig)
