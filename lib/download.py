import pandas as pd
from binance.client import Client
from tqdm import tqdm
client = Client()

def get_pair_tickers(symbol='USDT'):
    exchange_info = client.get_exchange_info()['symbols']
    symbols = [s['symbol'] for s in exchange_info]
    
    return [x for x in symbols if x.endswith(symbol)]


def download_symbol_data(symbol, start = "01 january 2018", interval = Client.KLINE_INTERVAL_1HOUR):
    klinesT = client.get_historical_klines(symbol, interval, start)
    df = pd.DataFrame(klinesT, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df = df.set_index(df['timestamp'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['timestamp']
    
    return df

def save_data_pickle(df, symbol):
    df.to_pickle(f'usdt_data/{symbol}.pickle')

def save_data_features_pickle(df, symbol):
    df.to_pickle(f'usdt_data_features/{symbol}.pickle')
    
def download_symbols_data(symbols):
    for symbol in tqdm(symbols):
        try:
            download_symbol_data(symbol)
        except:
            print(':)')

def update_data_symbol(symbol):
    lastindex = pd.read_pickle(f'usdt_data/{symbol}.pickle').index[-1]
    lastdata = download_symbol_data(symbol, start = str(lastindex))
    current_data = pd.read_pickle(f'usdt_data/{symbol}.pickle')
    newdf = pd.concat([current_data,lastdata],axis=0)
    
    return newdf

def update_data(symbols):
    for symbol in symbols:
            df = update_data_symbol(symbol)
            save_data_pickle(df, symbol)

def testImport():
    print("test")