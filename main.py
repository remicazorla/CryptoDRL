import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
import warnings
from binance.client import Client

client = Client()
warnings.simplefilter(action='ignore')   

def get_pair_tickers(symbol='USDT'):
    exchange_info = client.get_exchange_info()['symbols']
    symbols = [s['symbol'] for s in exchange_info]
    return [x for x in symbols if x.endswith(symbol)]

#adds TA to price dataframe from yfinance
def add_classic_indicators(df):
    np.seterr(invalid='ignore')
    # EMA
    df['ema5']=ta.trend.ema_indicator(close=df['close'], window=5)
    df['ema8']=ta.trend.ema_indicator(close=df['close'], window=8)
    df['ema12']=ta.trend.ema_indicator(close=df['close'], window=12)
    df['ema16']=ta.trend.ema_indicator(close=df['close'], window=16)
    df['ema20']=ta.trend.ema_indicator(close=df['close'], window=20)
    df['ema26']=ta.trend.ema_indicator(close=df['close'], window=26)
    
    # SMA
    df['sma5']=ta.trend.sma_indicator(close=df['close'], window=5)
    df['sma8']=ta.trend.sma_indicator(close=df['close'], window=8)
    df['sma12']=ta.trend.sma_indicator(close=df['close'], window=12)
    df['sma16']=ta.trend.sma_indicator(close=df['close'], window=16)
    df['sma20']=ta.trend.sma_indicator(close=df['close'], window=20)
    df['sma26']=ta.trend.sma_indicator(close=df['close'], window=26)
    
    # MACD
    macd2452 = ta.trend.MACD(close=df['close'], window_fast=24, window_slow=52, window_sign=9)
    macd1226 = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd_24_52'] = macd2452.macd_diff()
    df['macd_12_26'] = macd1226.macd_diff()

    # ADX
    df['adx7'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['close'], window = 14)
    df['adx10'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['close'], window = 10)
    df['adx14'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['close'], window = 14)
    df['adx20'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['close'], window = 20)
    df['adx30'] =ta.trend.adx(high=df['High'], low=df['Low'], close = df['close'], window = 30)
    
    # RSI
    df['rsi7'] = ta.momentum.RSIIndicator(close=df['close'], window=7).rsi()
    df['rsi10'] = ta.momentum.RSIIndicator(close=df['close'], window=10).rsi()
    df['rsi14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['rsi17'] = ta.momentum.RSIIndicator(close=df['close'], window=17).rsi()
    df['rsi20'] = ta.momentum.RSIIndicator(close=df['close'], window=20).rsi()
    df['rsi25'] = ta.momentum.RSIIndicator(close=df['close'], window=25).rsi()

    # STOCHASTIC RSI
    df['stochrsi7'] = ta.momentum.StochRSIIndicator(close=df['close'], window=7).stochrsi()
    df['stochrsi10'] = ta.momentum.StochRSIIndicator(close=df['close'], window=10).stochrsi()
    df['stochrsi14'] = ta.momentum.StochRSIIndicator(close=df['close'], window=14).stochrsi()
    df['stochrsi17'] = ta.momentum.StochRSIIndicator(close=df['close'], window=17).stochrsi()
    df['stochrsi20'] = ta.momentum.StochRSIIndicator(close=df['close'], window=20).stochrsi()
    df['stochrsi25'] = ta.momentum.StochRSIIndicator(close=df['close'], window=25).stochrsi()
    
    #filtrer avec EMA?
    #WilliamsR
    df['willr10'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['close'],lbp=10).williams_r()
    df['willr14'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['close'],lbp=14).williams_r()
    df['willr17'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['close'],lbp=17).williams_r()
    df['willr20'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['close'],lbp=20).williams_r()
    df['willr25'] = ta.momentum.WilliamsRIndicator(high=df['High'],low=df['Low'],close=df['close'],lbp=25).williams_r()
    
    # CCI
    df['CCI5'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['close'],window=5).cci()
    df['CCI7'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['close'],window=7).cci()
    df['CCI10'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['close'],window=10).cci()
    df['CCI14'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['close'],window=14).cci()
    df['CCI20'] = ta.trend.CCIIndicator(high=df['High'],low=df['Low'],close=df['close'],window=20).cci()

    #add talib features (volume, trend, momentum, volatility)
    ta_df = add_all_ta_features(df, open="Open", high="High", low="Low", close="close", volume="Volume")
    ta_df.index = ta_df.index.date
    return pd.concat([df,ta_df],axis=1).T.drop_duplicates().T

def get_custom_features(df):
    #prophet, arima, VAE, NLP, orderbook, fft, tsfresh, diffdiffdiff....
    
    #data_FT = df[['Date', 'GS']]
    #close_fft = np.fft.fft(np.asarray(data_FT['GS'].tolist()))
    #fft_df = pd.DataFrame({'fft':close_fft})
    #fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    #fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    #fft_list = np.asarray(fft_df['fft'].tolist())
    #for num_ in [2, 7, 15, 100]:
        #fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        #df['Fourier' + num_] = fft_list_m10
    return

def generate_lagged_variables(df):
    df['Returns n-1'] = df['Returns'].shift(1)
    df['Returns n-2'] = df['Returns'].shift(2)
    df['Returns n-3'] = df['Returns'].shift(3)
    df['Returns n-4'] = df['Returns'].shift(4)
    df['Returns n-5'] = df['Returns'].shift(5)
    df['Returns n-6'] = df['Returns'].shift(6)
    df['Returns n-7'] = df['Returns'].shift(7)
    df['Returns n-10'] = df['Returns'].shift(10)
    df['Returns n-15'] = df['Returns'].shift(15)
    df['Returns n-20'] = df['Returns'].shift(20)
    df['Returns n-25'] = df['Returns'].shift(25)
    df['RSI n-1'] = df['rsi14'].shift(1)
    df['RSI n-2'] = df['rsi14'].shift(2)
    df['RSI n-3'] = df['rsi14'].shift(3)
    df['RSI n-4'] = df['rsi14'].shift(4)
    df['RSI n-5'] = df['rsi14'].shift(5)
    df['RSI n-6'] = df['rsi14'].shift(6)
    df['RSI n-7'] = df['rsi14'].shift(7)
    return df

