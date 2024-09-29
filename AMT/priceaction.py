# C Import Libraries
import pandas as pd
from backtesting import Strategy, Backtest
from tpo_helper_intra import get_ticksize, get_rf, tpo
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
import datetime
from def_btxtra import btextra, equity_curve_plot, backtest_plot

df = pd.read_pickle(r'C:\Users\alex1\OneDrive\Public\NIFTY50-Minute.pkl')
split = int(len(df) / 2)
df_insample = df.head(split)
df_outsample = df.tail(split)

#df_to_test = df_insample.copy()
df_to_test = df_outsample.copy()

mode = 'tpo'
freq = 15
avglen = 30

ticksz = get_ticksize(df_outsample, freq=freq)
time1 = datetime.datetime.strptime('09:15', '%H:%M')
time2 = datetime.datetime.strptime('15:30', '%H:%M')
time_difference = time2 - time1
trading_hr = time_difference.seconds / 3600

df_to_test = get_rf(df_to_test.copy())
df_to_test['datetime'] = df_to_test.index

dfresample = df_to_test.resample(str(freq) + 'min').agg({'datetime': 'last', 'Open': 'first',
                                                           'High': 'max',
                                                           'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
dfresample = dfresample.dropna()

df_mp = tpo(dfresample.copy(), freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr)

df_mp_OHLC = df_mp[1][['Open', 'High', 'Low', 'Close', 'rf']]
df_mp_other = df_mp[1].drop(['rf'], axis=1)

# Convert to shift date
df_mp_OHLC['datetime'] = df_mp_other['datetime']
df_mp_OHLC['dateonly'] = df_mp_OHLC['datetime'].dt.date

df_mp_other = df_mp_other.set_index(df_mp_other['datetime'])
df_mp_open = df_mp_other.copy().resample('D').agg({
'Open': 'first',
'datetime': 'last'})

df_mp_daily = df_mp_other.copy().resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'datetime': 'last',
    'VAH': 'last',
    'POC': 'last',
    'VAL': 'last',
    'max_lvn': 'max',
    'min_lvn': 'min',
    'excess_high': 'last',
    'excess_low': 'last',
    'Volume': 'last',
    'TPO_Net': 'last'
})
# retain day open

df_mp_open['dateonly'] = df_mp_open['datetime'].dt.date
merged_df3 = pd.merge(df_mp_OHLC, df_mp_open, on='dateonly', how='left')

df_mp_daily = df_mp_daily.dropna()
df_mp_daily['POC_trend'] = np.where(df_mp_daily['POC'] > df_mp_daily['POC'].shift(1), 1, -1)
df_mp_daily['POC_small'] = df_mp_daily['POC'].rolling(20).mean()
df_mp_daily['POC_long'] = df_mp_daily['POC'].rolling(50).mean()
df_mp_daily['SMA_small'] = df_mp_daily['Close'].rolling(20).mean()
df_mp_daily['SMA_long'] = df_mp_daily['Close'].rolling(50).mean()

df_mp_daily['AR'] = (df_mp_daily['High']-df_mp_daily['Low']).rolling(20).mean()
df_mp_daily = df_mp_daily.dropna()
df_mp_daily['dateonly2'] = df_mp_daily['datetime'].dt.date
df_mp_daily['dateonly'] = df_mp_daily['dateonly2'].shift(-1)

merged_df2 = pd.merge(merged_df3, df_mp_daily, on='dateonly', how='left')
merged_df2 = merged_df2.dropna()
# rename the columns in consistent with our backtest
# merged_df2.rename(columns={'Open_y': 'DayOpen'}, inplace=True)
merged_df2 = merged_df2.drop('datetime', axis=1)
merged_df2.rename(columns={'Open_x': 'Open', 'High_x': 'High', 'Low_x': 'Low', 'Close_x': 'Close','Open_y': 'DayOpen','datetime_x': 'datetime'}, inplace=True)

merged_df2['date'] = merged_df2['datetime']  # this column needed for bt_extra function
merged_df2['rf_small'] = merged_df2['rf'].rolling(2).sum()
merged_df2['rf_long'] = merged_df2['rf'].rolling(5).sum()
merged_df2['High_5'] = merged_df2['High'].rolling(5).max()
merged_df2['Low_5'] = merged_df2['Low'].rolling(5).min()

merged_df2 = merged_df2.dropna()
# Visualization

# fig = tpo_fig(merged_df2.copy())
# plot(fig)

merged_df2['time'] = merged_df2['datetime'].dt.time
merged_df2['time'] = merged_df2['time'].astype(str)
merged_df2['time'] = merged_df2['time'].str.replace(':', '')
merged_df2['time'] = merged_df2['time'].astype(int)
merged_df2.set_index(merged_df2['datetime'], inplace=True)


merged_df2= merged_df2.dropna()
# conditions for long and short

long_condition = ("(merged_df2['High'] > merged_df2['High_5'].shift(1)) & "
                  # "(merged_df2['SMA_small'] > merged_df2['SMA_long']) &"
                  "(merged_df2['rf_small'] > merged_df2['rf_long']) &"
                 "(merged_df2['rf'] > merged_df2['rf'].shift(1)) &"
                 "(merged_df2['time'] >= 94400) & (merged_df2['time'] <= 132900)")

short_condition = ("(merged_df2['Low'] < merged_df2['Low_5'].shift(1)) &"
                   # "(merged_df2['SMA_small'] < merged_df2['SMA_long']) &"
                  "(merged_df2['rf'] < merged_df2['rf'].shift(1)) &"
                   "(merged_df2['rf_small'] < merged_df2['rf_long']) &"
                  "(merged_df2['rf'] < merged_df2['rf'].shift(1)) &"
                  "(merged_df2['time'] >= 94400) & (merged_df2['time'] <= 132900)")

merged_df2['long'] = np.where(eval(long_condition), 1, 0)
merged_df2['short'] = np.where(eval(short_condition), -1, 0)

class Hi5intra(Strategy):
    n1 = 1
    def init(self):
        pass

    def next(self):

        # Here bulk of the strategy is written
        if not self.position:
            if self.data.long[-1] == 1:
                self.buy()
                self.entry_price_buy = self.data.Close[-1]  # Set entry price

            elif self.data.short[-1] == -1:
                self.sell()
                self.entry_price_sell = self.data.Close[-1]  # Set entry price

        else:
            if self.position.is_long:
                if (self.data.time == 152900
                        # or (
                        #      self.data.Close[-1] > self.data.VAH[-1] and
                        #      self.data.Close[-2] < self.data.VAH[-2]
                        #     )
                        # # or (self.data.Low[-1] < self.data.Low_5[-2])
                        or (self.data.Low[-1] < self.entry_price_buy - (self.data.AR[-1]/self.n1))
                        # or (self.data.Close[-1] > self.entry_price_buy + (self.entry_price_buy * 0.01))

                ):
                    self.position.close()
            if self.position.is_short:
                if (self.data.time == 152900
                        # or (
                        #     self.data.Close[-1] < self.data.VAL[-1] and
                        #     self.data.Close[-2] > self.data.VAL[-2]
                        #     )
                        # # or (self.data.High[-1] > self.data.High_5[-2])
                        or (self.data.High[-1] > self.entry_price_sell + (self.data.AR[-1]/self.n1))
                        # or (self.data.Close[-1] < self.entry_price_sell - (self.entry_price_sell * 0.01))
                ):
                    self.position.close()

symbol = "NiftySpot"
backtest = Backtest(merged_df2, Hi5intra, cash=500_000,
                    trade_on_close=True, commission=0.0005)
stats = backtest.run()
print(stats)
trades = stats['_trades']

# part 2 visualisation
btreport = btextra(merged_df2.copy(), trades.copy(), freq)

print(btreport[0])

# C optimize
opt_stats = backtest.optimize(n1=range(1, 15),
                    maximize='Return [%]',
                              method='grid',   # skopt
                    constraint=lambda x: x.n1 >= 1,
                              return_heatmap=True,
                              random_state=2)
print(opt_stats)
best_n1 = opt_stats[0]._strategy.n1
print(f"The best n1 value is: {best_n1}")

# fig = btreport[1]
# plot(fig)
merged_df2.rename(columns={'SMA_small': 'indicator'}, inplace=True)
backtest_fig = backtest_plot(merged_df2, trades.copy(), indicator_plot= 'line',symbol=symbol)
plot(backtest_fig)

ec = stats['_equity_curve']
eqfig = equity_curve_plot(ec)
plot(eqfig)

# SO FAR V3 IS THE 2nd BEST and V5 is best add TPO >100 changed things
with open(f'backtestreport_{symbol}.txt', 'w') as f:
    f.write(str(stats) + '\n' + btreport[0] + '\n' + f'SL=0.01%. With commission 0.05% and no profit target\n Long Condition:'
                                                     f' {long_condition}+\nShort '
                                                     f'Condition'
                                                     f':{short_condition}')


# Equity Curve
