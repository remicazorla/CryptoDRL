import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
import warnings
from binance.client import Client
from alpha101 import add_artificial_variables


warnings.simplefilter(action='ignore')   

#adds TA to price dataframe from yfinance
def add_classic_indicators(df):
    np.seterr(invalid='ignore')
    c = df['close']
    l = df['low']
    h = df['high']
    
    # MACD
    macd2452 = ta.trend.MACD(close=c, window_fast=24, window_slow=52, window_sign=9)
    macd1226 = ta.trend.MACD(close=c, window_fast=12, window_slow=26, window_sign=9)
    df['macd_24_52'] = macd2452.macd_diff()
    df['macd_12_26'] = macd1226.macd_diff()

    for i in [5,8,12,16,20,26]:
        # EMA
        df[f'ema{i}']=ta.trend.ema_indicator(close=c, window=i)
        # SMA
        df[f'sma{i}']=ta.trend.sma_indicator(close=c, window=i)
    
    for i in [7,10,14,17,20,25]:
        # ADX
        df[f'adx{i}']=ta.trend.adx(high=h, low=l, close=c, window=i)
        # RSI
        df[f'rsi{i}']=ta.momentum.RSIIndicator(close=c, window=i).rsi()
        # STOCHRSI
        df[f'stochrsi{i}']=ta.momentum.StochRSIIndicator(close=c, window=i).stochrsi()
        # W%R
        df[f'willR{i}']=ta.momentum.WilliamsRIndicator(high=h,low=l,close=c,lbp=i).williams_r()
        # CCI
        df[f'cci{i}'] = ta.trend.CCIIndicator(high=h,low=l,close=c,window=i).cci()

    #add talib features (volume, trend, momentum, volatility)
    ta_df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
    ta_df.index = ta_df.index.date
    
    return pd.concat([df,ta_df],axis=1).T.drop_duplicates().T



def generate_lagged_variables(df):
    #tester le lag sur bcp de features
    for i in [1,2,3,4,5,6,7,10,15,20,25]:
        df[f'Returns n-{i}'] = df['Returns'].shift(i)
        df[f'rsi n-{i}'] = df['rsi14'].shift(i)
    return df

def get_custom_features(df):
    #alpha101
    df = add_artificial_variables(df)
    #prophet, arima, VAE, NLP, orderbook, fft, tsfresh, diffdiffdiff, GARCH, EWMA....
    
    #data_FT = df[['Date', 'GS']]
    #close_fft = np.fft.fft(np.asarray(data_FT['GS'].tolist()))
    #fft_df = pd.DataFrame({'fft':close_fft})
    #fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    #fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    #fft_list = np.asarray(fft_df['fft'].tolist())
    #for num_ in [2, 7, 15, 100]:
        #fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        #df['Fourier' + num_] = fft_list_m10
    return df