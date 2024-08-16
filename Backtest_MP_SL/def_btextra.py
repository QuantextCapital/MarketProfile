import pandas as pd
import numpy as np
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def btextra(df, trades, freq):
    # df = merged_df2.copy()

    # Merge based on nearest timestamp
    df['EntryTime'] = df['date']
    df['ExitTime'] = df['date']

    trades.index = trades['EntryTime']
    merged_entry = pd.merge_asof(trades, df, right_on='EntryTime', left_index=True,
                                 direction='nearest')
    trades_exit = trades.copy()
    trades_exit.index = trades_exit['ExitTime']
    merged_exit = pd.merge_asof(trades_exit, df, right_on='ExitTime', left_index=True,
                                direction='nearest')

    """
    This code snippet iterates over a list of trades and calculates the maximum notional 
    loss (Max Adverse Excursion) and maximum notional gain (Max Favorable Excursion) within a specific time range for each trade. 
    The `merged_df2` DataFrame is being sliced based on the entry and exit times of each trade.
     The `Low.min()` and `High.max()` functions are used to find the minimum and maximum values 
     of the `Low` and `High` columns within the specified time range. 
    The calculated values are then appended to the `maxloss` and `maxgain` lists respectively.
    Timedelta is used so it will exclude the entry and exit bar. 
    """
    maxloss = []
    maxgain = []
    for i in range(len(trades)):
        # i =1
        max_notional_loss = df[merged_entry.index[i] + datetime.timedelta(minutes=freq):merged_exit.index[i] - datetime.timedelta(
            minutes=freq)].Low.min()

        max_notional_gain = df[merged_entry.index[i] + datetime.timedelta(minutes=freq):merged_exit.index[i] - datetime.timedelta(
            minutes=freq)].High.max()

        maxloss.append(max_notional_loss)
        maxgain.append(max_notional_gain)
    trades['minlow'] = maxloss
    trades['maxhigh'] = maxgain

    """
    Whenever a trade is completed within 30 minutes, the corresponding value in the 
    `maxloss` and `maxgain` columns is set to 0 by replacing NaN.
    """

    trades['netmaxloss'] = np.where(trades['Size'] > 0, trades['minlow'] - trades['EntryPrice'],
                                    trades['EntryPrice'] - trades['maxhigh'])
    trades['netmaxgain'] = np.where(trades['Size'] > 0, trades['maxhigh'] - trades['EntryPrice'],
                                    trades['EntryPrice'] - trades['minlow'])
    trades['netmaxloss%'] = 100 * (trades['netmaxloss'] / trades['EntryPrice'])
    trades['netmaxgain%'] = 100 * (trades['netmaxgain'] / trades['EntryPrice'])

    trades['avg_pnl'] = trades['PnL'] / abs(trades['Size'])
    trades['avg_gain'] = np.where(trades['avg_pnl'] > 0, trades['avg_pnl'], 0)
    trades['avg_loss'] = np.where(trades['avg_pnl'] < 0, trades['avg_pnl'], 0)
    trades['avg_gain%'] = 100 * (trades['avg_gain'] / trades['EntryPrice'])
    trades['avg_loss%'] = 100 * (trades['avg_loss'] / trades['EntryPrice'])
    trades['position_long'] = np.where(trades['Size'] > 0, 1, 0)
    trades['position_short'] = np.where(trades['Size'] < 0, 1, 0)
    trades['position_profit'] = np.where(trades['PnL'] > 0, 1, 0)
    trades['position_loss'] = np.where(trades['PnL'] < 0, 1, 0)

    total_longs = trades['position_long'].sum()
    total_shorts = trades['position_short'].sum()
    total_wins = trades['position_profit'].sum()
    total_losses = trades['position_loss'].sum()

    avg_win = trades['avg_gain'].sum() / total_wins
    avg_loss = trades['avg_loss'].sum() / total_losses
    avg_winloss = abs(avg_win / avg_loss)

    trades['ConsecutiveWinners'] = trades.groupby((trades['position_profit'] != trades[
        'position_profit'].shift()).cumsum())['position_profit'].cumsum()

    trades['ConsecutiveLosers'] = trades.groupby((trades['position_loss'] !=
                                                  trades['position_loss'].shift()).cumsum(
    ))['position_loss'].cumsum()

    max_cons_winners = trades['ConsecutiveWinners'].max()
    max_cons_losers = trades['ConsecutiveLosers'].max()


    fill_values = {
        'maxhigh': 0,
        'minlow': 0,
        'netmaxgain': 0,
        'netmaxloss': 0,
        'netmaxgain%': 0,
        'netmaxloss%': 0
    }

    trades.fillna(value=fill_values, inplace=True)

    # plot scatterplot with max notional loss and profit in percentage
    trades['pnl_color'] = np.where(trades['PnL'] > 0, 'limegreen', 'magenta')
    trades['EntryTime'] = trades['EntryTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trades['EntryTime_str'] = trades['EntryTime'].astype(str)
    trades['EntryTime_str'] = '<br>' + trades['EntryTime_str']
    trades['Size_str'] = trades['Size'].astype(str)
    trades['Size_str'] = '<br>Size: ' + trades['Size_str']
    trades['netmaxloss_abs%'] = abs(trades['netmaxloss%']).round(2)
    trades['netmaxgain_abs%'] = abs(trades['netmaxgain%']).round(2)
    trades['netmaxloss_str%'] = '<br>NotionalLoss: ' + trades['netmaxloss_abs%'].astype(str) + '%'
    trades['netmaxgain_str%'] = '<br>NotionalGain: ' + trades['netmaxgain_abs%'].astype(str) + '%'
    trades['avg_gain_str%'] = (trades['avg_gain%'].round(2)).astype(str)
    trades['avg_gain_str%'] = '<br>Avg_Gain: ' + trades['avg_gain_str%'] + '%'
    trades['avg_loss_str%'] = (trades['avg_loss%'].round(2)).astype(str)
    trades['avg_loss_str%'] = '<br>Avg_Loss: ' + trades['avg_loss_str%'] + '%'

    avg_win_percentage = trades['avg_gain%'].sum() / total_wins
    avg_loss_percentage = trades['avg_loss%'].sum() / total_losses

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, row_heights=[1.0],
                        vertical_spacing=0.3, horizontal_spacing=0.3,
                        specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x=trades['netmaxloss_abs%'],
        y=trades['netmaxgain_abs%'],
        mode='markers',
        hovertext=trades['EntryTime_str'] + trades['Size_str'] + trades['avg_gain_str%'] + trades['avg_loss_str%'],
        marker=dict(color=trades['pnl_color'], symbol='circle'),
        name='max notional loss and profit in percentage',
        showlegend=False
    ))

    fig.update_xaxes(title='Notional_Loss%', showline=False, color='white', showgrid=False,
                     showticklabels=True,
                     rangeslider_visible=False,
                     tickangle=90)

    fig.update_yaxes(title='Notional_Profit%', color='white', showgrid=False,
                     showticklabels=True, row=1, col=1)

    fig.update_layout(title='NotionalLoss(MAE) vs NotionalProfit(MFE)', paper_bgcolor='black',
                      plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )

    stats_new = (f"Total_longs: {total_longs}\nTotal_Shorts: {total_shorts}\nAvg_Win:"
                 f" {round(avg_win, 2)}\nAvg_Loss: {round(avg_loss, 2)}\nAvg_Win%: "
                 f"{round(avg_win_percentage,2)}\n"
                 f"Avg_Loss%: {round(avg_loss_percentage,2)}\n"
                 f"Avg_Win/Loss: {round(avg_winloss, 2)}\n"
                 f"Max_Consecutive_Winners: {max_cons_winners}\n"
                 f"Max_Consecutive_Losers: {max_cons_losers}\n")

    return stats_new, fig

def equity_curve_plot(df):
    df['DrawdownPct'] = 100 * df['DrawdownPct']
    df['color'] = np.where(df['DrawdownPct'] < 1, 'green', 'red')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Equity Curve vs Drawdown'), row_heights=[0.70, 0.30],
                        vertical_spacing=0.003,
                        horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.Equity,
        mode='lines',
        line=dict(color='cyan',
                  width=4,
                  # dash = 'dot'
                  ),
        # hovertext=hovertext,
        # hoverinfo='text',
        showlegend=False
    ),
        secondary_y=False, col=1, row=1)

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df.DrawdownPct,
        mode='lines+markers',
        line=dict(color='gray', width=1),
        marker=dict(color=df['color'], size=4),
        name='DD',
        showlegend=False
    ),
        secondary_y=False, col=1, row=2)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=1)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=2)

    fig.update_yaxes(title='EquityCurve', color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=1, col=1, tickformat=',d')

    fig.update_yaxes(title='DrawDown%', color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=2, col=1)

    fig.update_layout(title='EquityCurve vs DD', paper_bgcolor='black', plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )
    return fig

def backtest_plot(df, trades):
    # merged_list = [group[1] for group in df.groupby(df.index.date)]
    # df = merged_df2.copy()
    trades['color'] = np.where(trades.Size > 0, 'limegreen', 'magenta')
    trades['color'] = trades['color'].astype(str)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30],
                        vertical_spacing=0.003, horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 showlegend=False,
                                 name="NiftySpot", opacity=0.3), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['indicator'],
            name='DailyStrength',
            showlegend=False,
            marker=dict(
                # color=df['color_power'],  # Using the column values for color
                colorscale='Viridis',  # Specify the desired color scale
            )
        ),
        secondary_y=False, col=1, row=2)

    # for df in merged_list:
    #     hovertext_vah = (f"VAH: {df.iloc[0]['VAH']}<br>"
    #                      )
    #
    #     fig.add_trace(go.Scatter(
    #         x=[df.iloc[0]['date'], df.iloc[-1]['date']],
    #         y=[df.iloc[0]['VAH'], df.iloc[0]['VAH']],
    #         mode='lines',
    #         line=dict(color='green',
    #                   width=1,
    #                   dash='dot'),
    #         hovertext=hovertext_vah,
    #         hoverinfo='text',
    #         showlegend=False
    #     ))
    #
    #     hovertext_val = (f"VAL: {df.iloc[0]['VAL']}<br>"
    #                      )
    #
    #     fig.add_trace(go.Scatter(
    #         x=[df.iloc[0]['date'], df.iloc[-1]['date']],
    #         y=[df.iloc[0]['VAL'], df.iloc[0]['VAL']],
    #         mode='lines',
    #         line=dict(color='red',
    #                   width=1,
    #                   dash='dot'),
    #         hovertext=hovertext_val,
    #         hoverinfo='text',
    #         showlegend=False
    #     ))

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
                      # dash = 'dot'
                      ),
            hovertext=hovertext_trades,
            hoverinfo='text',
            showlegend=False
        ))

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=1)

    fig.update_xaxes(showline=False, color='white', showgrid=False, showticklabels=False,
                     type='category', rangeslider_visible=False,
                     tickangle=90, zeroline=False, col=1, row=2)

    fig.update_yaxes(color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=1, col=1, tickformat=',d')

    fig.update_yaxes(color='white', showgrid=False,
                     zeroline=False, showticklabels=True, row=2, col=1, tickformat=',d')

    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black',
                      autosize=True, uirevision=True
                      )
    return fig
