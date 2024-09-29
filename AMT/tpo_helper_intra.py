# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:26:04 2020

@author: alex1
"""
import math
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objs as go


# import itertools


def get_ticksize(data, freq=30):
    # data = df
    numlen = int(len(data) / 2)
    # sample size for calculating tick size = 50% of most recent data
    tztail = data.tail(numlen).copy()
    tztail['tz'] = tztail.Close.rolling(freq).std()  # std. dev of 30 period rolling
    tztail = tztail.dropna()
    ticksize = np.ceil(tztail['tz'].mean() * 0.50)  # 1/2  of mean std. dev is our tick size

    if ticksize < 0.2:
        ticksize = 0.2  # minimum ticksize limit

    return int(ticksize)


def abc(dft_rs, session_hr=6.5, freq=30):
    caps = [' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' J', ' K', ' L', ' M',
            ' N', ' O', ' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z']
    abc_lw = [x.lower() for x in caps]
    Aa = caps + abc_lw
    alimit = math.ceil(session_hr * (60 / freq)) + 3
    alimit2 = len(dft_rs)
    if alimit < alimit2:
        alimit = alimit2
    else:
        alimit = alimit

    if alimit > 52:
        alphabets = Aa * int(
            (np.ceil((
                                 alimit - 52) / 52)) + 1)  # if bar frequency is less than 30 minutes then multiply list
    else:
        alphabets = Aa[0:alimit]
    bk = [28, 31, 35, 40, 33, 34, 41, 44, 35, 52, 41, 40, 46, 27, 38]
    ti = []
    for s1 in bk:
        ti.append(Aa[s1 - 1])
    tt = (''.join(ti))

    return alphabets, tt


def tpo(df, freq=30, ticksize=10, style='tpo', session_hr=6.5):
    # df = dfresample.copy()
    try:
        DFcontext = [group[1] for group in df.groupby(df.index.date)]
        df2_list = []
        dftpo_list = []
        for c in range(len(DFcontext)):
            # c = 1
            df2 = DFcontext[c].copy()
            if len(df2) > int(60 / freq):
                df2 = df2.drop_duplicates('datetime')
                df2 = df2.reset_index(inplace=False, drop=True)
                # df2['rol_mx'] = df2['High'].cummax()
                # df2['rol_mn'] = df2['Low'].cummin()
                # df2['ext_up'] = df2['rol_mn'] > df2['rol_mx'].shift(2)
                # df2['ext_dn'] = df2['rol_mx'] < df2['rol_mn'].shift(2)

                alphabets = abc(df2, session_hr, freq)[0]
                alphabets = alphabets[0:len(df2)]
                hh = df2['High'].max()
                ll = df2['Low'].min()
                # day_range = hh - ll
                df2['abc'] = alphabets
                # place represents total number of steps to take to compare the TPO count
                place = int(np.ceil((hh - ll) / ticksize))
                # kk = 0
                abl_bg = []
                tpo_countbg = []
                pricel = []
                volcountbg = []
                # datel = []
                for u in range(place):
                    abl = []
                    tpoc = []
                    volcount = []
                    p = ll + (u * ticksize)
                    for lenrs in range(len(df2)):
                        if p >= df2['Low'][lenrs] and p < df2['High'][lenrs]:
                            abl.append(df2['abc'][lenrs])
                            tpoc.append(1)
                            volcount.append((df2['Volume'][lenrs]) / freq)
                    abl_bg.append(''.join(abl))
                    tpo_countbg.append(sum(tpoc))
                    volcountbg.append(sum(volcount))
                    pricel.append(p)

                dftpo = pd.DataFrame({'close': pricel, 'alphabets': abl_bg,
                                      'tpocount': tpo_countbg, 'volsum': volcountbg})
                # drop empty rows
                # dftpo['alphabets'].replace('', np.nan, inplace=True)
                dftpo.replace({'alphabets', ''}, np.nan, inplace=True)
                dftpo = dftpo.dropna()
                # dftpo = dftpo.reset_index(inplace=False, drop=True)

                dftpo_sort = dftpo.copy().sort_values(by='tpocount', ascending=False).reset_index(drop=True)
                poc = dftpo_sort['close'][0]

                total_tpo = dftpo_sort['tpocount'].sum()

                # Calculate the volume for the value area (70% of the total volume)
                value_area_tpo = total_tpo * 0.7

                # Determine the Value Area High (VAH) and Value Area Low (VAL)
                cumulative_tpo = 0
                vah = val = poc  # Initialize to POC as a starting point

                for index, row in dftpo_sort.iterrows():
                    # print(index)
                    # print(row)
                    cumulative_tpo += row['tpocount']
                    if cumulative_tpo >= value_area_tpo / 2 and row['close'] < poc:
                        val = row['close']
                        break

                cumulative_tpo = 0

                for index, row in dftpo_sort.iterrows():
                    # print(index)
                    # print(row)
                    cumulative_tpo += row['tpocount']
                    if cumulative_tpo >= value_area_tpo / 2 and row['close'] > poc:
                        vah = row['close']
                        break

                df2['VAH'] = round(vah, 2)
                df2['POC'] = round(poc, 2)
                df2['VAL'] = round(val, 2)
                df2['TPO_Net'] = total_tpo

                tpoval = dftpo[ticksize:-(ticksize)]['tpocount']  # take mid section
                exhandle_index = np.where(tpoval <= 2, tpoval.index, None)  # get index where TPOs are 2
                exhandle_index = list(filter(None, exhandle_index))
                distance = ticksize * 3  # distance b/w two ex handles / lvn

                lvn_list = []
                for ex in exhandle_index[0:-1:distance]:
                    lvn_list.append(dftpo['close'][ex])

                if len(lvn_list)>=2:
                    max_lvn = max(lvn_list)
                    min_lvn = min(lvn_list)
                if len(lvn_list)==1:
                    max_lvn = lvn_list[0]
                    min_lvn = lvn_list[0]
                if len(lvn_list) == 0:
                    max_lvn = 0
                    min_lvn = 0

                df2['max_lvn'] = max_lvn
                df2['min_lvn'] = min_lvn

                # divide dftpo at poc index
                poc_index = dftpo[dftpo['close'] == poc].index[0]
                dftpo_post_poc = dftpo.iloc[poc_index:]
                dftpo_pre_poc = dftpo.iloc[:poc_index]
                top_tpo_count = dftpo_post_poc[dftpo_post_poc['tpocount'] == 1]
                bottom_tpo_count = dftpo_pre_poc[dftpo_pre_poc['tpocount'] == 1]
                excess_high = top_tpo_count['tpocount'].sum()
                excess_low = bottom_tpo_count['tpocount'].sum()

                df2['excess_high'] = excess_high
                df2['excess_low'] = excess_low

                df2_list.append(df2)
                dftpo_list.append(dftpo)


            else:
                print('not enough bars for date {}'.format(df2['datetime'][0]))
                dftpo = pd.DataFrame()
                df2 = pd.DataFrame()
                df2_list.append(df2)
                dftpo_list.append(dftpo)
                pass
        df2_concat = pd.concat(df2_list)
        dftpo_concat = pd.concat(dftpo_list)

    except Exception as e:
        print(e)
        dftpo = pd.DataFrame()
        dftpo_list.append(dftpo)
        df2_concat = pd.DataFrame()
        dftpo_concat = pd.DataFrame()

        pass

    return dftpo_concat, df2_concat


# !!! fetch all MP derived results here with date and do extra context analysis


# def get_context(df_hi, freq=30, ticksize=5, style='tpo', session_hr=6.5):
#     """
#     df_hi: resampled DataFrame
#     mean_val: mean dily values
#
#     return: 1) list of dataframes with TPOs 2) DataFrame with ranking for bubbles 3) DataFrame with a ranking breakdown for hover text
#     """
#     #    df_hi=dfresample.copy()
#     #   df_hi = dflive.copy()
#     # df_hi = df.copy()
#     try:
#
#         DFcontext = [group[1] for group in df_hi.groupby(df_hi.index.date)]
#         dfmp_l = []
#         i_poctpo_l = []
#         i_tposum = []
#         vah_l = []
#         poc_l = []
#         val_l = []
#         bt_l = []
#         lvn_l = []
#         # excess_l = []
#         date_l = []
#         volume_l = []
#         rf_l = []
#         open_l = []
#         close_l = []
#         hh_l = []
#         ll_l = []
#         range_l = []
#
#         for c in range(len(DFcontext)):  # c=0, ticksize=ticksz, style=mode for testing
#
#             dfc1 = DFcontext[c].copy()
#             if len(dfc1) > int(60 / freq):
#                 dfc1.iloc[:, 2:6] = dfc1.iloc[:, 2:6].apply(pd.to_numeric)
#
#                 dfc1 = dfc1.reset_index(inplace=False, drop=True)
#                 mpc = tpo(dfc1, freq, ticksize, style, session_hr)
#                 dftmp = mpc['df']
#                 dfmp_l.append(dftmp)
#                 # for day types
#                 i_poctpo_l.append(dftmp['tpocount'].max())
#                 i_tposum.append(dftmp['tpocount'].sum())
#                 # !!! get value areas
#                 vah_l.append(mpc['vah'])
#                 poc_l.append(mpc['poc'])
#                 val_l.append(mpc['val'])
#
#                 bt_l.append(mpc['bal_target'])
#                 lvn_l.append(mpc['lvn'])
#                 # excess_l.append(mpc['excess'])
#
#                 # !!! operatio of non profile stats
#                 date_l.append(dfc1.datetime[0])
#                 open_l.append(dfc1.iloc[0]['Open'])
#                 close_l.append(dfc1.iloc[-1]['Close'])
#                 ll_l.append(dfc1.High.max())
#                 hh_l.append(dfc1.Low.min())
#                 range_l.append(dfc1.High.max() - dfc1.Low.min())
#
#                 volume_l.append(dfc1.Volume.sum())
#                 rf_l.append(dfc1.rf.sum())
#                 # !!! get IB
#                 dfc1['cumsumvol'] = dfc1.Volume.cumsum()
#                 dfc1['cumsumrf'] = dfc1.rf.cumsum()
#                 dfc1['cumsumhigh'] = dfc1.High.cummax()
#                 dfc1['cumsummin'] = dfc1.Low.cummin()
#                 # !!! append ib values
#                 # 60 min = 1 hr divide by time frame to get number of bars
#
#         # dffin = pd.concat(dfcon_l)
#         # max_po = max(i_poctpo_l)
#         # min_po = min(i_poctpo_l)
#
#         dist_df = pd.DataFrame({'date': date_l, 'Open': open_l, 'High': hh_l, 'Low': ll_l, 'Close':
#             close_l,
#                                 'maxtpo': i_poctpo_l, 'tpocount': i_tposum, 'VAH': vah_l,
#                                 'POC': poc_l, 'VAL': val_l, 'Bal_target': bt_l, 'lvnlist': lvn_l,
#                                 'Volume': volume_l, 'rfd': rf_l, 'ranged': range_l})
#
#     except Exception as e:
#         print(str(e))
#         ranking_df = []
#         dfmp_l = []
#         dist_df = []
#
#     return (dfmp_l, dist_df)


# def get_dayrank(dist_df, mean_val):
#     """
#      It calculates final power ranking based on various factors such as
#      1 Total number of low volume nodes or single prints if using price only inside the profile
#      body,
#
#      2. Tpo distribution factor, If the number of TPOs above the poc are higher than the number of
#       TPOs below then it assigns positive value else, it assigns negative value
#
#      3.Day type-  the daytypes it considers are
#        a. trend day which has the highest score of 4
#        b. Trend distribution day  which has 2nd highest score of 3
#        c.  Normal variation day which has 3rd highest score of 2
#        d,  Neutral day, which has the lowest score of one
#
#     4.  If current day is poc is higher than previous days POC then it assigns positive value else it assigns negative value
#
#     5 If current day is value area high is higher than previous day, is value area high? then it assigns positive value else it assigns negative value
#
#     6 If current day is value area low is higher than previous day, is value area low? then it
#     assigns positive value else it assigns negative value
#
#     7.  Similarly, it considers relationship between current days Close and previous days
#     Close, VAL, VAH and POC
#
#     8  It calculates the mean value for the rotation factor, if the value is +ve, then it assigns
#     positive value and else it assigns negative value.
#      accordingly.
#
#      Finally, it converts these numbers into percentage so we can easily  get the picture of the
#      total strength of the day.
#       This percentage can be in <unk>and positive if the percentage is negative, then it assigns
#       the color magenta.Otherwise it assigns the color lime green.
#       Which is useful while displaying it on the graph. Thanks to the flexibility of plotly library
#
#     """
#
#     # LVNs
#     # dist_df = df_distribution.copy()
#
#     lvnlist = dist_df['lvnlist'].to_list()
#     cllist = dist_df['Close'].to_list()
#     lvn_powerlist = []
#     total_lvns = 0
#     for c, llist in zip(cllist, lvnlist):
#         if len(llist) == 0:
#             delta_lvn = 0
#             total_lvns = 0
#             # print('total_lvns 0 ')
#             lvn_powerlist.append(total_lvns)
#         else:
#             for l in llist:
#                 delta_lvn = c - l
#                 if delta_lvn >= 0:
#                     lvn_i = 1
#                 else:
#                     lvn_i = -1
#                 total_lvns = total_lvns + lvn_i
#             lvn_powerlist.append(total_lvns)
#         total_lvns = 0
#
#     dist_df['Single_Prints'] = lvn_powerlist
#
#     dist_df['distr'] = dist_df.tpocount / dist_df.maxtpo
#     dismean = math.floor(dist_df.distr.mean())
#     dissig = math.floor(dist_df.distr.std())
#
#     # Assign day types based on TPO distribution and give numerical value for each day types for calculating total strength at the end
#
#     dist_df['daytype'] = np.where(np.logical_and(dist_df.distr >= dismean,
#                                                  dist_df.distr < dismean + dissig),
#                                   'Trend Distribution Day', '')
#
#     dist_df['daytype_num'] = np.where(np.logical_and(dist_df.distr >= dismean,
#                                                      dist_df.distr < dismean + dissig), 3, 0)
#
#     dist_df['daytype'] = np.where(np.logical_and(dist_df.distr < dismean,
#                                                  dist_df.distr >= dismean - dissig),
#                                   'Normal Variation Day',
#                                   dist_df['daytype'])
#
#     dist_df['daytype_num'] = np.where(np.logical_and(dist_df.distr < dismean,
#                                                      dist_df.distr >= dismean - dissig), 2,
#                                       dist_df['daytype_num'])
#
#     dist_df['daytype'] = np.where(dist_df.distr < dismean - dissig,
#                                   'Neutral Day', dist_df['daytype'])
#
#     dist_df['daytype_num'] = np.where(dist_df.distr < dismean - dissig,
#                                       1, dist_df['daytype_num'])
#
#     dist_df['daytype'] = np.where(dist_df.distr > dismean + dissig,
#                                   'Trend Day', dist_df['daytype'])
#     dist_df['daytype_num'] = np.where(dist_df.distr > dismean + dissig,
#                                       4, dist_df['daytype_num'])
#     dist_df['daytype_num'] = np.where(dist_df.Close >= dist_df.POC, dist_df.daytype_num * 1,
#                                       dist_df.daytype_num * -1)  # assign signs as per bias
#
#     daytypes = dist_df['daytype'].to_list()
#
#     # volume comparison with mean
#     rf_mean = mean_val['rf_mean']
#     vol_mean = mean_val['volume_mean']
#
#     dist_df['vold_zscore'] = (dist_df.Volume - vol_mean) / dist_df.Volume.std(ddof=0)
#     dist_df['rfd_zscore'] = (abs(dist_df.rfd) - rf_mean) / abs(dist_df.rfd).std(ddof=0)
#     a, b = 1, 4
#     x, y = dist_df.rfd_zscore.min(), dist_df.rfd_zscore.max()
#     dist_df['norm_rf'] = (dist_df.rfd_zscore - x) / (y - x) * (b - a) + a
#
#     p, q = dist_df.vold_zscore.min(), dist_df.vold_zscore.max()
#     dist_df['norm_volume'] = (dist_df.vold_zscore - p) / (q - p) * (b - a) + a
#
#     dist_df['Volume_Factor'] = np.where(dist_df.Close >= dist_df.POC, dist_df.norm_volume * 1,
#                                         dist_df.norm_volume * -1)
#     dist_df['Rotation_Factor'] = np.where(dist_df.rfd >= 0, dist_df.norm_rf * 1,
#                                           dist_df.norm_rf * -1)
#
#     # !!! get ranking based on distribution data frame aka dist_df
#     ranking_df = dist_df.copy()
#     ranking_df['VAH_vs_yVAH'] = np.where(ranking_df.VAH >= ranking_df.VAH.shift(), 1, -1)
#     ranking_df['VAL_vs_yVAL'] = np.where(ranking_df.VAL >= ranking_df.VAL.shift(), 1, -1)
#     ranking_df['POC_vs_yPOC'] = np.where(ranking_df.POC >= ranking_df.POC.shift(), 1, -1)
#     ranking_df['H_vs_yH'] = np.where(ranking_df.High >= ranking_df.High.shift(), 1, -1)
#     ranking_df['L_vs_yL'] = np.where(ranking_df.Low >= ranking_df.Low.shift(), 1, -1)
#     ranking_df['Close_vs_yCL'] = np.where(ranking_df.Close >= ranking_df.Close.shift(), 1, -1)
#     ranking_df['CL>POC<VAH'] = np.where(
#         np.logical_and(ranking_df.Close >= ranking_df.POC,
#                        ranking_df.Close < ranking_df.VAH), 1, 0)
#     ranking_df['CL<poc>val'] = np.where(
#         np.logical_and(ranking_df.Close < ranking_df.POC,
#                        ranking_df.Close >= ranking_df.VAL), -1,
#         0)  # Max is 2
#     ranking_df['CL<VAL'] = np.where(ranking_df.Close < ranking_df.VAL, -2, 0)
#     ranking_df['CL>=VAH'] = np.where(ranking_df.Close >= ranking_df.VAH, 2, 0)
#     # check if value in column ranking_df['Volume_Factor']  is Nan
#     ranking_df['Volume_Factor'] = np.where(ranking_df['Volume_Factor'].isnull(), 0,
#                                            ranking_df['Volume_Factor'])
#     # check if value in column ranking_df['Rotation_Factor']  is Nan
#     ranking_df['Rotation_Factor'] = np.where(ranking_df['Rotation_Factor'].isnull(), 0,
#                                              ranking_df['Rotation_Factor'])
#     # check if value in column ranking_df['daytype_num']  is Nan
#     ranking_df['daytype_num'] = np.where(ranking_df['daytype_num'].isnull(), 0,
#                                          ranking_df['daytype_num'])
#
#     ranking_df['power1'] = 100 * (
#             (
#                         ranking_df.VAH_vs_yVAH + ranking_df.VAL_vs_yVAL + ranking_df.POC_vs_yPOC + ranking_df.H_vs_yH +
#                         ranking_df.L_vs_yL + ranking_df['Close_vs_yCL'] + ranking_df['CL>POC<VAH'] +
#                         ranking_df[
#                             'CL<poc>val'] + ranking_df.Single_Prints +
#                         ranking_df['CL<VAL'] + ranking_df[
#                             'CL>=VAH'] + ranking_df.Volume_Factor + ranking_df.Rotation_Factor + ranking_df.daytype_num) / 14)
#
#     c, d = 25, 100
#     r, s = abs(ranking_df.power1).min(), abs(ranking_df.power1).max()
#     ranking_df['power'] = (abs(ranking_df.power1) - r) / (s - r) * (d - c) + c
#     ranking_df = ranking_df.round(2)
#     # ranking_df['power'] = abs(ranking_df['power1'])
#
#     breakdown_df = ranking_df.copy()[
#         ['Single_Prints', 'daytype_num', 'Volume_Factor', 'Rotation_Factor', 'VAH_vs_yVAH',
#          'VAL_vs_yVAL', 'POC_vs_yPOC', 'H_vs_yH',
#          'L_vs_yL', 'Close_vs_yCL', 'CL>POC<VAH', 'CL<poc>val', 'CL<VAL', 'CL>=VAH']].transpose()
#
#     breakdown_df = breakdown_df.round(2)
#
#     return (ranking_df, breakdown_df)
#

def get_rf(df):
    df['cup'] = np.where(df['Close'] >= df['Close'].shift(), 1, -1)
    df['hup'] = np.where(df['High'] >= df['High'].shift(), 1, -1)
    df['lup'] = np.where(df['Low'] >= df['Low'].shift(), 1, -1)

    df['rf'] = df['cup'] + df['hup'] + df['lup']
    df = df.drop(['cup', 'lup', 'hup'], axis=1)
    return df


# def get_mean(data, avglen=30, freq=30):
#     """
#     data: pandas dataframe 1 min frequency
#     avglen: Length for mean values
#     freq: timeframe for the candlestick & TPOs
#
#     return: a) daily mean for volume, rotational factor (absolute value),  b) session length
#
#     """
#
#     dfhist = get_rf(data.copy())
#     dfhistd = dfhist.resample("D").agg(
#         {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
#          'Volume': 'sum',
#          'rf': 'sum', })
#     dfhistd = dfhistd.dropna()
#     comp_days = len(dfhistd)
#
#     vm30 = dfhistd['Volume'].rolling(avglen).mean()
#     volume_mean = vm30.iloc[len(vm30) - 1]
#     rf30 = abs((dfhistd['rf'])).rolling(avglen).mean()  # it is abs mean to get meaningful value
#     # to compare daily values
#     rf_mean = rf30.iloc[len(rf30) - 1]
#
#     mask = dfhist.index[0] < dfhist.index[-1]
#     dfsession = dfhist.loc[mask]
#     session_hr = math.ceil(len(dfsession) / 60)
#     # dfib = df.head(int(60/freq))
#     # dfib['Volume'].plot()
#
#     all_val = dict(volume_mean=volume_mean, rf_mean=rf_mean, session_hr=session_hr)
#
#     return all_val

def tpo_fig(dfall):
    # dfall = dfmp_values.copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('any name'), row_heights=[0.100, 0.0], vertical_spacing=0.003,
                        horizontal_spacing=0.0003,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])

    fig.add_trace(go.Candlestick(x=dfall.datetime,
                                 open=dfall['Open'],
                                 high=dfall['High'],
                                 low=dfall['Low'],
                                 close=dfall['Close'],
                                 showlegend=False,
                                 name="AnyName", opacity=0.3), row=1, col=1)

    dfall.set_index('datetime', inplace=True)
    DFcontext = [group[1] for group in dfall.groupby(dfall.index.date)]

    for df in DFcontext:

        hovertext_vah = (f"VAH: {df.iloc[0]['VAH']}<br>"
                         )
        fig.add_trace(go.Scatter(
             x=[df.index[0], df.index[-1]],
             y=[df.iloc[0]['VAH'], df.iloc[-1]['VAH']],
             mode='lines',
             line=dict(color='green',
                       width=1,
                       dash = 'dot'
                       ),
             hovertext=hovertext_vah,
             hoverinfo='text',
             showlegend=False
         ),
         secondary_y=False, col=1, row=1)


        hovertext_val = (f"VAL: {df.iloc[0]['VAL']}<br>"
                         )
        fig.add_trace(go.Scatter(
             x=[df.index[0], df.index[-1]],
             y=[df.iloc[0]['VAL'], df.iloc[-1]['VAL']],
             mode='lines',
             line=dict(color='red',
                       width=1,
                       dash = 'dot'
                       ),
             hovertext=hovertext_val,
             hoverinfo='text',
             showlegend=False
         ),
         secondary_y=False, col=1, row=1)

        hovertext_poc = (f"POC: {df.iloc[0]['POC']}<br>"
                         f"Excess_high: {df.iloc[0]['excess_high']}<br>"
                         f"Excess_low: {df.iloc[0]['excess_low']}<br>")
        fig.add_trace(go.Scatter(
             x=[df.index[0], df.index[-1]],
             y=[df.iloc[0]['POC'], df.iloc[-1]['POC']],
             mode='lines',
             line=dict(color='yellow',
                       width=1,
                       dash = 'dot'
                       ),
             hovertext=hovertext_poc,
             hoverinfo='text',
             showlegend=False
         ),
         secondary_y=False, col=1, row=1)

        hovertext_maxlvn = (f"Max_LVN: {df.iloc[0]['max_lvn']}<br>"
                         )
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[df.iloc[0]['max_lvn'], df.iloc[-1]['max_lvn']],
            mode='lines',
            line=dict(color='white',
                      width=1,
                      dash='dot'
                      ),
            hovertext=hovertext_maxlvn,
            hoverinfo='text',
            showlegend=False
        ),
            secondary_y=False, col=1, row=1)

        hovertext_minlvn = (f"Min_LVN: {df.iloc[0]['min_lvn']}<br>"
                         )
        fig.add_trace(go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[df.iloc[0]['min_lvn'], df.iloc[-1]['min_lvn']],
            mode='lines',
            line=dict(color='white',
                      width=1,
                      dash='dot'
                      ),
            hovertext=hovertext_minlvn,
            hoverinfo='text',
            showlegend=False
        ),
            secondary_y=False, col=1, row=1)

        fig.update_xaxes(title='date',showline=False, color='white', showgrid=False, showticklabels=False,
                         type='category',rangeslider_visible=False,
                            tickangle=90, zeroline=False, row=1,col=1)

        fig.update_yaxes(title='price', color='white', showgrid=False,
                            zeroline=False, showticklabels=True,row=1,col=1, tickformat = ',d')

        fig.update_layout(title='TPO Profile',paper_bgcolor='black',plot_bgcolor='black',
                          autosize=True,uirevision=True
                          )

    return fig
