# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 02:56:32 2020

@author: alex1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 06:34:11 2020

@author: alex1
"""

import pandas as pd
import plotly.graph_objects as go
import dash  # (version 1.12.0) pip install dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from tpo_helper_simple import (get_ticksize, abc, get_mean, get_rf, get_context, get_dayrank)
import numpy as np
from datetime import timedelta
import yfinance as yf

app = dash.Dash(__name__)

# ticksz = 5
# ib_start,ib_end = ('9:15', '10:15')
# trading_hr = 7

refresh_int = 15  # refresh interval in seconds for live updates
freq = 30
avglen = 5  # num days mean to get values
days_to_display = avglen  # Number of last n days you want on the screen to display
mode = 'tpo'  # for volume --> 'vol'

ticker = "BTC-USD"  #"^NSEI"
dfhist = yf.download(tickers=ticker, period=str(avglen) + 'd', interval='1m')

# To get the TPO chart from local data change the file path to your csv file"
# filePath = 'd:/anaconda/Scripts/niftyf.csv'
# dfhist = pd.read_csv(filePath, header=0)
# dfhist['DateTime'] = pd.to_datetime(dfhist['DateTime'], format='%Y-%m-%d %H:%M:%S')
# dfhist = dfhist.set_index('DateTime', drop=True, inplace=False)

#  1 min historical data in symbol,datetime,open,high,low,close,volume. Header names not needed

ticksz = get_ticksize(dfhist, freq=freq)

# !!! create seperate col for pandas dtime as plotly accepts strings. pd dtime used for pd resample
mean_val = get_mean(dfhist, avglen=avglen, freq=freq)
trading_hr = mean_val['session_hr']

# !!! get rotational factor
dfhist = get_rf(dfhist.copy())
# !!! resample to desire time frequency. For TPO charts 30 min is optimal
dfhist['datetime'] = dfhist.index

dfhist = dfhist.resample(str(freq) + 'min').agg(
    {'datetime': 'first', 'Open': 'first', 'High': 'max',
     'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
dfhist = dfhist.dropna()

# slice df based on days_to_display parameter
dt1 = dfhist.index[-1]
sday1 = dt1 - timedelta(days_to_display)
dfhist = dfhist[(dfhist.index.date > sday1.date())]


# !!! concat current data to avoid insufficient bar num error

def live_merge(dfli):
    """
    dfli: pandas dataframe with live quotes.

    This is the live data, and will continue to refresh. Since it merges with historical data keep the format same though source
    can be different..
    For this we only need small sample and if there are duplicate quotes duplicate values will get droped keeping the original value.
    """

    dflive = get_rf(dfli.copy())
    dflive['datetime'] = dflive.index
    dflive = dflive.resample(str(freq) + 'min').agg(
        {'datetime': 'first', 'Open': 'first', 'High': 'max',
         'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'rf': 'sum'})
    dflive = dflive.dropna()
    df_final = pd.concat([dfhist, dflive])
    # df_final = df_final.reset_index(inplace=False, drop=True)
    df_final = df_final.drop_duplicates()

    return (df_final)


# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div(
    html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Link('For questions, ping me on Twitter', href='https://youtube.com/@quantext'),
        html.Br(),
        dcc.Link('FAQ and python source code', href='http://www.github.com/quantextcapital'),
        html.H4('@QuantextCcapital'),
        dcc.Graph(id='quantext'),
        dcc.Interval(
            id='interval-component',
            interval=refresh_int * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output(component_id='quantext', component_property='figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    """
    main loop for refreshing the data and to display the chart. It gets triggered every n second
     as per our
    settings.
    """

    # !!! for datetime
    # dfli = get_data(mode = 'compact')
    dfli = yf.download(tickers=ticker, period='1d', interval='1m')
    df = live_merge(dfli)

    # df.iloc[:,2:] = df.iloc[:,2:].apply(pd.to_numeric)

    # !!! soplit the dataframe with new date
    DFList = [group[1] for group in df.groupby(df.index.date)]
    # !!! for context based bubbles at the top with text hovers
    dfcontext = get_context(df, freq=freq, ticksize=ticksz, style=mode, session_hr=trading_hr)
    #  get market profile DataFrame and ranking as a series for each day.
    # @todo: IN next version, display the ranking DataFrame with drop-down menu
    dfmp_list = dfcontext[0]
    df_distribution = dfcontext[1]
    df_ranking = get_dayrank(df_distribution.copy(), mean_val)
    ranking = df_ranking[0]

    power1 = ranking.power1  # Non-normalised day's strength
    power = ranking.power  # Normalised day's strength for dynamic shape size for markers
    breakdown = df_ranking[1]
    dh_list = ranking.highd
    dl_list = ranking.lowd

    # !!! get context based on IB As of now it only considers IB volume and IB price relation to previous day's value area
    # IB is 1st 1 hour of the session. Not useful for scrips with global 24 x 7 session

    fig = go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         showlegend=True,
                                         name=ticker,
                                         opacity=0.3)])

    # !!! get TPO for each day
    for i in range(len(dfmp_list)):  # test the loop with i=1

        df1 = DFList[
            i].copy()
        df_mp = dfmp_list[i]
        irank = ranking.iloc[i]
        # df_mp['i_date'] = df1['datetime'][0]
        df_mp['i_date'] = irank.date
        # # @todo: background color for text
        df_mp['color'] = np.where(
            np.logical_and(df_mp['close'] > irank.vallist, df_mp['close'] < irank.vahlist), 'green',
            'white')

        df_mp = df_mp.set_index('i_date', inplace=False)

        fig.add_trace(
            go.Scatter(x=df_mp.index, y=df_mp.close, mode="text", name=str(df_mp.index[0]),
                       text=df_mp.alphabets,
                       showlegend=False, textposition="top right",
                       textfont=dict(family="verdana", size=6, color=df_mp.color)))
        if power1[i] < 0:
            my_rgb = 'rgba({power}, 3, 252, 0.5)'.format(power=abs(165))
        else:
            my_rgb = 'rgba(23, {power}, 3, 0.5)'.format(power=abs(252))

        brk_f_list_maj = []
        f = 0
        for f in range(len(breakdown.columns)):
            brk_f_list_min = []
            for index, rows in breakdown.iterrows():
                brk_f_list_min.append(index + str(': ') + str(rows[f]) + '<br />')
            brk_f_list_maj.append(brk_f_list_min)

        breakdown_values = ''  # for bubbles
        for st in brk_f_list_maj[i]:
            breakdown_values += st

        fig.add_trace(go.Scatter(
            x=[irank.date],
            y=[df['High'].max()],
            mode="markers",
            marker=dict(color=my_rgb, size=0.90 * power[i],
                        line=dict(color='rgb(17, 17, 17)', width=2)),
            hovertext=[
                '<br />Insights:<br />VAH:  {}<br /> POC:  {}<br /> VAL:  {}<br /> Balance Target:  {}<br /> Day Type:  {}<br />strength: {}<br />BreakDown:  {}<br />{}<br />{}'.format(
                    irank.vahlist,
                    irank.poclist, irank.vallist, irank.btlist, irank.daytype, irank.power, '',
                    '-------------------', breakdown_values)], showlegend=False))


        lvns = irank.lvnlist

        for lvn in lvns:
            fig.add_shape(
                # Line Horizontal
                type="line",
                x0=df1.iloc[0]['datetime'],
                y0=lvn,
                x1=df1.iloc[5]['datetime'],
                y1=lvn,
                line=dict(
                    color="darksalmon",
                    width=2,
                    dash="dashdot", ), )

        # day high and low
        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=df1.iloc[0]['datetime'],
            y0=dl_list[i],
            x1=df1.iloc[0]['datetime'],
            y1=dh_list[i],
            line=dict(
                color="gray",
                width=1,
                dash="dashdot",),)


    # @todo: last price marker. Color code as per close above poc or below
    ltp = df1.iloc[-1]['Close']
    if ltp >= irank.poclist:
        ltp_color = 'green'
    else:
        ltp_color = 'red'

    fig.add_trace(go.Scatter(
        x=[df1.iloc[-1]['datetime']],
        y=[df1.iloc[-1]['Close']],
        mode="text",
        name="last traded price",
        text=['last ' + str(df1.iloc[-1]['Close'])],
        textposition="bottom right",
        textfont=dict(size=9, color=ltp_color),
        showlegend=False
    ))

    fig.layout.xaxis.color = 'white'
    fig.layout.yaxis.color = 'white'
    fig.layout.autosize = True
    fig["layout"]["height"] = 800
    # fig.layout.hovermode = 'x'
    # fig.layout.plot_bgcolor = '#44494C'
    # fig.layout.paper_bgcolor = '#44494C'

    fig.update_xaxes(title_text='Time', title_font=dict(size=18, color='white'),
                     tickangle=45, tickfont=dict(size=8, color='white'), showgrid=False,
                     dtick=len(dfmp_list))

    fig.update_yaxes(title_text=ticker, title_font=dict(size=18, color='white'),
                     tickfont=dict(size=12, color='white'), showgrid=False)
    # ??? which is the right method?
    #    fig.update_layout(template="plotly_dark", title = "@"+abc()[1],autosize=True,
    #                      xaxis = dict(showline = True, color = 'white'), yaxis = dict(showline = True, color = 'white'))

    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["xaxis"]["tickformat"] = "%H:%M:%S"
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False, type='category',
                     showticklabels=False, color='Black',
                     # showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                     tickangle=0)
    fig.update_layout(paper_bgcolor='black', plot_bgcolor='black',
                      autosize=True, uirevision=True,
                      )

    return fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(port=8050, host='127.0.0.1', debug=True)
    # app.run_server(port=8050, host='0.0.0.0', debug=True)
