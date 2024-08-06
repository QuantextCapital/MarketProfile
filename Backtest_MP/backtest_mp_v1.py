import pandas as pd
from backtesting import Strategy, Backtest
from tpo_helper_simple import get_ticksize, get_mean, get_rf, get_context, get_dayrank
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np

df = pd.read_pickle(r'C:\Users\alex1\OneDrive\Public\NIFTY50-Minute.pkl')
split = int(len(df)/2)
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

dfresample = df_outsample.resample(str(freq)+'min').agg({'datetime': 'last', 'Open': 'first',
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
                         'tpocount', 'VAH', 'POC', 'VAL','ranged', 'Single_Prints','daytype_num','power1']]
merged_df2.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'rf','tpocount',
                      'VAH', 'POC', 'VAL','ranged', 'Single_Prints','daytype_num','power1']
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
                              (merged_df2['power1'] > 5) & (merged_df2['power1'] < 80) &
                              (merged_df2['time'] >= 95900) & (merged_df2['time'] <= 142900)
                              ,
                              1, 0)

merged_df2['short'] = np.where((merged_df2['Close'] < merged_df2['VAH']) &
                                (merged_df2['Close'].shift(1) > merged_df2['VAH']) &
                               (merged_df2['power1'] < -5) & (merged_df2['power1'] > -80) &
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
            elif self.data.short[-1] == -1:
                self.sell()
        else:
            if self.position.is_long:
                if self.data.time == 152900 or (self.data.Close[-1] > self.data.VAH[-1] and
                                                self.data.Close[-2] < self.data.VAH[-2]):
                    self.position.close()
            if self.position.is_short:
                if self.data.time == 152900 or (self.data.Close[-1] < self.data.VAL[-1] and
                                                self.data.Close[-2] > self.data.VAL[-2]):
                    self.position.close()

backtest = Backtest(merged_df2, EightyPercent, cash=50_000, trade_on_close=True)
stats = backtest.run()
print(stats)

# part 2 visualisation
trades = stats['_trades']
trades['color'] = np.where(trades.Size > 0, 'limegreen', 'magenta')
trades['color'] = trades['color'].astype(str)

merged_df2['color_power'] = np.where(merged_df2['power1']>0,'green','red')
merged_df2['color_power'] = merged_df2['color_power'].astype(str)

merged_list = [group[1] for group in merged_df2.groupby(merged_df2.index.date)]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30],
                    vertical_spacing=0.003,horizontal_spacing=0.0003,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

fig.add_trace(go.Candlestick(x=merged_df2.index,
                                     open=merged_df2['Open'],
                                     high=merged_df2['High'],
                                     low=merged_df2['Low'],
                                     close=merged_df2['Close'],
                                     showlegend=False,
                                     name="NiftySpot", opacity=0.3), row=1, col=1)
fig.add_trace(
    go.Bar(
        x=merged_df2.index,
        y=merged_df2['power1'],
        name='DailyStrength',
        showlegend=False,
        marker=dict(
            color=merged_df2['color_power'],  # Using the column values for color
            #colorscale='Viridis',  # Specify the desired color scale
        )
    ),
    secondary_y=False, col=1, row=2)

for df in merged_list:
    hovertext_vah = (f"VAH: {df.iloc[0]['VAH']}<br>"
)

    fig.add_trace(go.Scatter(
        x=[df.iloc[0]['date'], df.iloc[-1]['date']],
        y=[df.iloc[0]['VAH'], df.iloc[0]['VAH']],
        mode='lines',
        line=dict(color='green',
                  width=1,
                  dash = 'dot'),
        hovertext=hovertext_vah,
        hoverinfo='text',
        showlegend=False
    ))

    hovertext_val = (f"VAL: {df.iloc[0]['VAL']}<br>"
                     )

    fig.add_trace(go.Scatter(
        x=[df.iloc[0]['date'], df.iloc[-1]['date']],
        y=[df.iloc[0]['VAL'], df.iloc[0]['VAL']],
        mode='lines',
        line=dict(color='red',
                  width=1,
                  dash='dot'),
        hovertext=hovertext_val,
        hoverinfo='text',
        showlegend=False
    ))

for j in range(len(trades)):
    hovertext_trades = (f"EntryPrice: {trades.iloc[j]['EntryPrice']}<br>"
                 f"Size: {trades.iloc[j]['Size']}<br>"
                 f"ExitPrice: {trades.iloc[j]['ExitPrice']}<br>"
                 f"PnL: {trades.iloc[j]['PnL']}")

    fig.add_trace(go.Scatter(
        x=[trades.iloc[j]['EntryTime'], trades.iloc[j]['ExitTime']],
        y=[trades.iloc[j]['EntryPrice'], trades.iloc[j]['ExitPrice']],
        mode='lines',
        line=dict(color=trades['color'][j],
                  width=3,
                  #dash = 'dot'
                  ),
        hovertext=hovertext_trades,
        hoverinfo='text',
        showlegend=False
    ))

fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                 type='category',rangeslider_visible=False,
                    tickangle=90, zeroline=False, col=1,row=1)

fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                 type='category',rangeslider_visible=False,
                    tickangle=90, zeroline=False, col=1,row=2)

fig.update_yaxes(color='white', showgrid=False,
                    zeroline=False, showticklabels=True,row=1,col=1, tickformat = ',d')

fig.update_yaxes(color='white', showgrid=False,
                    zeroline=False, showticklabels=True,row=2,col=1, tickformat = ',d')

fig.update_layout(paper_bgcolor='black',plot_bgcolor='black',
                  autosize=True,uirevision=True
                  )

plot(fig)
