import pandas as pd
import numpy as np
import time

ohlcv = ['open', 'high', 'low', 'close', 'volume']

def rank(df):
    return df.rank(pct=True)

def scale(df):
    return df.div(df.abs().sum(), axis=0)

def log(df):
    return np.log1p(df)

def sign(df):
    return np.sign(df)

def power(df, exp):
    return df.pow(exp)

def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
    return df.shift(t)

def ts_delta(df, period=1):
    return df.diff(period)

def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return df.rolling(window).sum()

def ts_mean(df, window=10):
    return df.rolling(window).mean()

def WMA(Data, period):
    weighted = []
    for i in range(len(Data)):
            try:
                total = np.arange(1, period + 1, 1) # weight matrix
                matrix = Data[i - period + 1: i + 1, 3:4]
                matrix = np.ndarray.flatten(matrix)
                matrix = total * matrix # multiplication
                wma = (matrix.sum()) / (total.sum()) # WMA
                weighted = np.append(weighted, wma) # add to array
            except ValueError:
                pass
    return weighted

def ts_weighted_mean(df, period=10):
    return (df.apply(lambda x: WMA(x, timeperiod=period)))

def ts_std(df, window=10):
    return (df.rolling(window).std())

def ts_rank(df, window=10):
    return (df.rolling(window).apply(lambda x: x.rank().iloc[-1]))

def ts_product(df, window=10):
    return (df .rolling(window).apply(np.prod))

def ts_min(df, window=10):
        return df.rolling(window).min()
    
def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax).add(1)

def ts_argmin(df, window=10):
    return (df.rolling(window).apply(np.argmin).add(1))

def ts_corr(x, y, window=10):
    return x.rolling(window).corr(y)

def ts_cov(x, y, window=10):
    return x.rolling(window).cov(y)

def add_artificial_variables(df):
    #https://github.com/stefan-jansen/machine-learning-for-trading/blob/main/24_alpha_factor_library/03_101_formulaic_alphas.ipynb

    vwap = df['open'].add(df['high']).add(df['low']).add(df['close']).div(4)
    o = df['open'].copy()
    h = df['high'].copy()
    l = df['low'].copy()
    c = df['close'].copy()
    r = df['returns'].copy()
    v = df['volume'].copy()

    s1 = rank(ts_delta(log(df['volume']), 2))
    s2 = rank((df['close'] / df['open']) - 1)
    adv20 = ts_mean(v, 20)

    sma2 = ts_mean(c, 2)
    sma8 = ts_mean(c, 8)
    std8 = ts_std(c, 8)
    cond_1 = sma8.add(std8) < sma2
    cond_2 = sma8.add(std8) > sma2
    cond_3 = v.div(ts_mean(v, 20)) < 1
    
    #Alpha 001
#     c[r < 0] = ts_std(r , 20)
#     df['alpha1'] = (rank(ts_argmax(power(c, 2), 5)).mul(-.5))
    
    #Alpha 002

    alpha = -ts_corr(s1, s2, 6)
    df['alpha2'] = alpha.replace([-np.inf, np.inf], np.nan)
    #Alpha 003
    df['alpha3'] = (-ts_corr(rank(df['open']), rank(df['volume']), 10).replace([-np.inf, np.inf], np.nan))


    #Alpha 004
#     df['alpha4'] = (-ts_rank(rank(df['low']), 9))

    #Alpha 005
    df['alpha5'] = (rank(df['open'].sub(ts_mean(vwap, 10))).mul(rank(df['close'].sub(vwap)).mul(-1).abs()))
    #Alpha 006
    df['alpha6'] = (-ts_corr(o, v, 10))
    #Alpha 008
    df['alph8'] = (-(rank(((ts_sum(o, 5) * ts_sum(r, 5)) -
                       ts_lag((ts_sum(o, 5) * ts_sum(r, 5)), 10)))))
    #Alpha 009
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 5) > 0,
                             close_diff.where(ts_max(close_diff, 5) < 0,
                                              -close_diff))
    df['alpha9'] = (alpha)
    #Alpha 010
    close_diff = ts_delta(c, 1)
    alpha = close_diff.where(ts_min(close_diff, 4) > 0,
                             close_diff.where(ts_min(close_diff, 4) > 0,
                                              -close_diff))
    df['alpha10'] = (rank(alpha))
    #Alpha 011
    df['alpha11'] = (rank(ts_max(vwap.sub(c), 3)).add(rank(ts_min(vwap.sub(c), 3))).mul(rank(ts_delta(v, 3))))
    #Alpha 012
    df['alpha12'] = (sign(ts_delta(v, 1)).mul(-ts_delta(c, 1)))
    #Alpha 014
    alpha = -rank(ts_delta(r, 3)).mul(ts_corr(o, v, 10).replace([-np.inf,np.inf],np.nan))
    df['alpha14'] = (alpha)
    #Alpha 015
    alpha = (-ts_sum(rank(ts_corr(rank(h), rank(v), 3).replace([-np.inf, np.inf], np.nan)), 3))
    df['alpha15'] = (alpha)
    #Alpha 016
    df['alpha16'] = (-rank(ts_cov(rank(h), rank(v), 5)))
    
    #Alpha 017

#     df['alpha17'] = (-rank(ts_rank(c, 10)).mul(rank(ts_delta(ts_delta(c, 1), 1))).mul(rank(ts_rank(v.div(adv20), 5))))

    #Alpha 018
    df['alpha18'] = (-rank(ts_std(c.sub(o).abs(), 5).add(c.sub(o)).add(ts_corr(c, o, 10).replace([-np.inf,np.inf],np.nan))))
    #Alpha 019
    df['alpha19'] = (-sign(ts_delta(c, 7) + ts_delta(c, 7)).mul(1 + rank(1 + ts_sum(r, 250))))
    #Alpha 020
    df['alpha20'] = (rank(o - ts_lag(h, 1)).mul(rank(o - ts_lag(c, 1))).mul(rank(o - ts_lag(l, 1))).mul(-1))
    #Alpha 021


    alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3],choicelist=[-1, 1, -1], default=1),index=c.index)
    df['alpha21'] = (alpha)
    #Alpha 022
    df['alpha22'] = (ts_delta(ts_corr(h, v, 5).replace([-np.inf,np.inf],np.nan), 5).mul(rank(ts_std(c, 20))).mul(-1))
    #Alpha 023
    df['alpha23'] = (ts_delta(h, 2).mul(-1).where(ts_mean(h, 20) < h, 0))
    #Alpha 024
    cond = ts_delta(ts_mean(c, 100), 100) / ts_lag(c, 100) <= 0.05
    df['alpha24'] = (c.sub(ts_min(c, 100)).mul(-1).where(cond, -ts_delta(c, 3)))
    #Alpha 025
    df['alpha25'] = (rank(-r.mul(adv20).mul(vwap).mul(h.sub(c))))
    
#     #Alpha 026
#     df['alpha26'] = (ts_max(ts_corr(ts_rank(v, 5), ts_rank(h, 5), 5).replace([-np.inf, np.inf], np.nan), 3).mul(-1))

    #Alpha 027
    cond = rank(ts_mean(ts_corr(rank(v),rank(vwap), 6), 2))
    alpha = cond.notnull().astype(float)
    df['alpha27'] = (alpha.where(cond <= 0.5, -alpha))
    #Alpha 028
    df['alpha28'] = (scale(ts_corr(adv20, l, 5).replace([-np.inf, np.inf], 0).add(h.add(l).div(2).sub(c))))
    
    #Alpha 029
#     df['alpha29'] = (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-rank(ts_delta((c - 1), 5)))), 2))))), 5).add(ts_rank(ts_lag((-1 * r), 6), 5)))

    #Alpha 030
    close_diff = ts_delta(c, 1)
    df['alpha30'] = (rank(sign(close_diff).add(sign(ts_lag(close_diff, 1))).add(sign(ts_lag(close_diff, 2))))
            .mul(-1).add(1)
            .mul(ts_sum(v, 5))
            .div(ts_sum(v, 20)))
    #Alpha 032
    df['alpha32'] = (scale(ts_mean(c, 7).sub(c))
            .add(20 * scale(ts_corr(vwap,
                                    ts_lag(c, 5), 230))))
    #Alpha 033
    df['alpha33'] = (rank(o.div(c).mul(-1).add(1).mul(-1)))
    #Alpha 034
    df['alpha34'] = (rank(rank(ts_std(r, 2).div(ts_std(r, 5))
                      .replace([-np.inf, np.inf],
                               np.nan))
                 .mul(-1)
                 .sub(rank(ts_delta(c, 1)))
                 .add(2)))

    #Alpha 035
#     df['alpha35'] = (ts_rank(v, 32)
#             .mul(1 - ts_rank(c.add(h).sub(l), 16))
#             .mul(1 - ts_rank(r, 32)))

    #Alpha 036
#     df['alpha36'] = (rank(ts_corr(c.sub(o), ts_lag(v, 1), 15)).mul(2.21)
#             .add(rank(o.sub(c)).mul(.7))
#             .add(rank(ts_rank(ts_lag(-r, 6), 5)).mul(0.73))
#             .add(rank(abs(ts_corr(vwap, adv20, 6))))
#             .add(rank(ts_mean(c, 200).sub(o).mul(c.sub(o))).mul(0.6)))

    #Alpha 037
    df['alpha37'] = (rank(ts_corr(ts_lag(o.sub(c), 1), c, 200))
            .add(rank(o.sub(c))))
    

    #Alpha 038
#     df['alpha38'] = (rank(ts_rank(o, 10))
#             .mul(rank(c.div(o).replace([-np.inf, np.inf], np.nan)))
#             .mul(-1))


    #Alpha 040
    df['alpha40'] = (rank(ts_std(h, 10))
            .mul(ts_corr(h, v, 10))
            .mul(-1))

    
    #Alpha 041
    df['alpha41'] = (power(h.mul(l), 0.5)
            .sub(vwap))


    #Alpha 042
    df['alpha42'] = (rank(vwap.sub(c))
            .div(rank(vwap.add(c))))
    
    #Alpha 043
#     df['alpha43'] = (ts_rank(v.div(adv20), 20)
#             .mul(ts_rank(ts_delta(c, 7).mul(-1), 8)))

    #Alpha 044
    df['alpha44'] = (ts_corr(h, rank(v), 5)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1))

    
    #Alpha 045
    df['alpha45'] = (rank(ts_mean(ts_lag(c, 5), 20))
            .mul(ts_corr(c, v, 2)
                 .replace([-np.inf, np.inf], np.nan))
            .mul(rank(ts_corr(ts_sum(c, 5),
                              ts_sum(c, 20), 2)))
            .mul(-1))

  

    #Alpha 047
    df['alpha47'] = (rank(c.pow(-1)).mul(v).div(adv20)
            .mul(h.mul(rank(h.sub(c))
                       .div(ts_mean(h, 5)))
                 .sub(rank(ts_delta(vwap, 5)))))

    #Alpha 048
    #df['alpha48'] = (indneutralize(((ts_corr(ts_delta(c, 1), ts_delta(ts_lag(c, 1), 1), 250) * 
    #   ts_delta(c, 1)) / c), IndClass.subindustry) / 
    #   ts_sum(((ts_delta(c, 1) / ts_lag(c, 1))^2), 250))
    
    #Alpha 049
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
            .sub(ts_delta(c, 10).div(10)) >= -0.1 * c)
    df['alpha49'] = (-ts_delta(c, 1)
            .where(cond, 1))

 
    
    #Alpha 050
    df['alpha50'] = (ts_max(rank(ts_corr(rank(v),
                                rank(vwap), 5)), 5)
            .mul(-1))

    
    #Alpha 51
    cond = (ts_delta(ts_lag(c, 10), 10).div(10)
           .sub(ts_delta(c, 10).div(10)) >= -0.05 * c)
    df['alpha51'] = (-ts_delta(c, 1)
            .where(cond, 1))
    
    #Alpha 052
#     df['alpha52'] = (ts_delta(ts_min(l, 5), 5)
#             .mul(rank(ts_sum(r, 240)
#                       .sub(ts_sum(r, 20))
#                       .div(220)))
#             .mul(ts_rank(v, 5)))

    
    
    #Alpha 053
    df['alpha53'] = (ts_delta(h.sub(c)
                     .mul(-1).add(1)
                     .div(c.sub(l)
                          .add(1e-6)), 9)
            .mul(-1))


    
    
    #Alpha 054
    df['alpha54'] = (l.sub(c).mul(o.pow(5)).mul(-1)
            .div(l.sub(h).replace(0, -0.0001).mul(c ** 5)))


    #Alpha 055
    df['alpha55'] = (ts_corr(rank(c.sub(ts_min(l, 12))
                         .div(ts_max(h, 12).sub(ts_min(l, 12))
                              .replace(0, 1e-6))),
                    rank(v), 6)
            .replace([-np.inf, np.inf], np.nan)
            .mul(-1))


        
    #Alpha 060
#     df['alpha60'] = (scale(rank(c.mul(2).sub(l).sub(h)
#                        .div(h.sub(l).replace(0, 1e-5))
#                        .mul(v))).mul(2)
#             .sub(scale(rank(ts_argmax(c, 10)))).mul(-1))


    #Alpha 061
    df['alpha61'] = (rank(vwap.sub(ts_min(vwap, 16)))
            .lt(rank(ts_corr(vwap, ts_mean(v, 180), 18)))
            .astype(int))


    
    #Alpha 062
    df['alpha62'] = (rank(ts_corr(vwap, ts_sum(adv20, 22), 9))
            .lt(rank(
                rank(o).mul(2))
                .lt(rank(h.add(l).div(2))
                    .add(rank(h))))
            .mul(-1))

    #Alpha 064
    w = 0.178404
    df['alpha64'] = (rank(ts_corr(ts_sum(o.mul(w).add(l.mul(1 - w)), 12),
                         ts_sum(ts_mean(v, 120), 12), 16))
            .lt(rank(ts_delta(h.add(l).div(2).mul(w)
                               .add(vwap.mul(1 - w)), 3)))
            .mul(-1))

    #Alpha 065
    w = 0.00817205
    df['alpha65'] = (rank(ts_corr(o.mul(w).add(vwap.mul(1 - w)),
                         ts_mean(ts_mean(v, 60), 9), 6))
            .lt(rank(o.sub(ts_min(o, 13))))
            .mul(-1))


    #Alpha 068
#     w = 0.518371
#     df['alpha68'] = (ts_rank(ts_corr(rank(h), rank(ts_mean(v, 15)), 9), 14)
#             .lt(rank(ts_delta(c.mul(w).add(l.mul(1 - w)), 1)))
#             .mul(-1))

    #Alpha 074
    w = 0.0261661
    df['alpha74'] = (rank(ts_corr(c, ts_mean(ts_mean(v, 30), 37), 15))
            .lt(rank(ts_corr(rank(h.mul(w).add(vwap.mul(1 - w))), rank(v), 11)))
            .mul(-1))

    

    #Alpha 075
    df['alpha75'] = (rank(ts_corr(vwap, v, 4))
            .lt(rank(ts_corr(rank(l), rank(ts_mean(v, 50)), 12)))
            .astype(int))



    #Alpha 078
    w = 0.352233
    df['alpha78'] = (rank(ts_corr(ts_sum((l.mul(w).add(vwap.mul(1 - w))), 19),
                         ts_sum(ts_mean(v, 40), 19), 6))
            .pow(rank(ts_corr(rank(vwap), rank(v), 5))))


 

    #Alpha 081
#     df['alpha81'] = (rank(log(ts_product(rank(rank(ts_corr(vwap,
#                                                   ts_sum(ts_mean(v, 10), 50), 8))
#                                      .pow(4)), 15)))
#             .lt(rank(ts_corr(rank(vwap), rank(v), 5)))
#             .mul(-1))

    #Alpha 083
    s = h.sub(l).div(ts_mean(c, 5))
    df['alpha83'] = (rank(rank(ts_lag(s, 2))
                 .mul(rank(rank(v)))
                 .div(s).div(vwap.sub(c).add(1e-3)))
            .replace((np.inf, -np.inf), np.nan))
    
   
    #Alpha 084
#     df['alpha84'] = (rank(power(ts_rank(vwap.sub(ts_max(vwap, 15)), 20),
#                        ts_delta(c, 6))))

    
    #Alpha 085
#     w = 0.876703
#     df['alpha85'] = (rank(ts_corr(h.mul(w).add(c.mul(1 - w)), ts_mean(v, 30), 10))
#             .pow(rank(ts_corr(ts_rank(h.add(l).div(2), 4),
#                               ts_rank(v, 10), 7))))

    #Alpha 086
#     df['alpha86'] = (ts_rank(ts_corr(c, ts_mean(ts_mean(v, 20), 15), 6), 20)
#             .lt(rank(c.sub(vwap)))
#             .mul(-1))

    
#     #Alpha 094
#     df['alpha94'] = (rank(vwap.sub(ts_min(vwap, 11)))
#             .pow(ts_rank(ts_corr(ts_rank(vwap, 20),
#                                  ts_rank(ts_mean(v, 60), 4), 18), 2))
#             .mul(-1))

    #Alpha 095
#     df['alpha95'] = (rank(o.sub(ts_min(o, 12)))
#             .lt(ts_rank(rank(ts_corr(ts_mean(h.add(l).div(2), 19),
#                                      ts_sum(ts_mean(v, 40), 19), 13).pow(5)), 12))
#             .astype(int))

    #Alpha 099
    df['alpha99'] = ((rank(ts_corr(ts_sum((h.add(l).div(2)), 19),
                          ts_sum(ts_mean(v, 60), 19), 8))
             .lt(rank(ts_corr(l, v, 6)))
             .mul(-1)))

    #Alpha 101
    df['alpha101'] = (c.sub(o).div(h.sub(l).add(1e-3)))

    return df