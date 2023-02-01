from setup import PATH_FEATURE,PATH_DATA,PATH_SYMBOL,PATH_BAD_SYMBOL,NBR_VALUE_FEATURE,DEBUG_DL,TIMER_DL
from datetime import date, timedelta
from lib import indicators as indi 
from binance.client import Client
from lib import timerLib as timeL
import concurrent.futures
import pandas as pd
import time 
import csv
import os 

client = Client()

#update_data_features for global update of data and features

#get tickers from binance, symbol = USDT
def get_pair_tickers(symbol='USDT'):    
    exchange_info = client.get_exchange_info()['symbols']
    symbols = [s['symbol'] for s in exchange_info]
    return [x for x in symbols if x.endswith(symbol)]

#get tickers in dataBase
def get_symbol(sort_binance = True):
    try:
        symbols = []
        with open(PATH_SYMBOL, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                symbols.append(row[0])
        return list(set(symbols))
    except:
        folder_path = "data/usdt_data_features"
        files = os.listdir(folder_path)
        file_names = [file.split(".")[0] for file in files if file.endswith(".pickle")]
        if(sort_binance):
            return [ticker for ticker in get_pair_tickers() if ticker in file_names]
        else:
            return list(set(file_names))

def get_bad_symbol(sort_binance = True):
    try:
        symbols = []
        with open(PATH_BAD_SYMBOL, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                symbols.append(row[0])
        return list(set(symbols))
    except:
        folder_path = "data/usdt_data_features"
        files = os.listdir(folder_path)
        file_names = [file.split(".")[0] for file in files if file.endswith(".pickle")]
        if(sort_binance):
            return [ticker for ticker in get_pair_tickers() if not ticker in file_names]
        else:
            return list(set(file_names))

def save_symbol(symbols = get_symbol()):
    try:
        if not os.path.isfile(PATH_SYMBOL):
            open(PATH_SYMBOL, "w").close()

        with open(PATH_SYMBOL, "a", newline="\n") as f:
            writer = csv.writer(f)
            if isinstance(symbols, str):
                if not symbols in get_bad_symbol():
                    writer.writerow([symbols])
            else:
                for sym in symbols:
                    if not sym in get_symbol():
                        writer.writerow([sym])
    except:
        print("Impossible d'enregistrer les symboles")

def save_bad_symbol(symbols):
    try:
        if not os.path.isfile(PATH_BAD_SYMBOL):
            open(PATH_BAD_SYMBOL, "w").close()

        with open(PATH_BAD_SYMBOL, "a", newline="\n") as f:
            writer = csv.writer(f)
            if isinstance(symbols, str):
                if not symbols in get_bad_symbol():
                    writer.writerow([symbols])
            else:
                for sym in symbols:
                    if not sym in get_bad_symbol():
                        writer.writerow([sym])
    except:
        print("Impossible d'enregistrer les mauvais symboles")

#download data from binance with start offset
def download_symbol_data(symbol, start = "01 january 2018", interval = Client.KLINE_INTERVAL_1HOUR):
    try:
        klinesT = client.get_historical_klines(symbol, interval, start)
        df = pd.DataFrame(klinesT, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df = df.set_index(df['timestamp'])
        df.index = pd.to_datetime(df.index, unit='ms')
        del df['timestamp']
    except:
        print(f"ERROR | Dowload impossible : {symbol}")
        return False
        
    return df
#update all features in database
def update_data_features(symbols = get_symbol()):
    start = time.time()
    cmpt = 1
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(update_data_features_symbol, symbol) for symbol in symbols]
        
        for f in concurrent.futures.as_completed(results):
            symbol = symbols[results.index(f)]
            try:
                f.result()
                if cmpt % 10 == 0 and TIMER_DL:
                    timediff = timeL.get_time_from_ms(timeL.diff(start))
                    timeNeeded = timeL.get_time_needed(start, cmpt, len(symbols))
                    print(f"\n \t###   {cmpt} / {len(symbols)}  |  {timediff[2]}h:{timediff[1]}m:{timediff[0]}s  | {timeNeeded[2]}h:{timeNeeded[1]}m:{timeNeeded[0]}s  ###\t\n")
            except Exception as e:
                print(f"Update Symbol : {symbol} Failed : {e}")
            cmpt+=1

#update data from binance for a single symbol    
def update_data_symbol(symbol):
    try:
        if(os.path.isfile(f'{PATH_DATA}/{symbol}.pickle')):
            df =  pd.read_pickle(f'{PATH_DATA}/{symbol}.pickle')
            start = str(df.index[-1])
            lastdata = download_symbol_data(symbol, start = start)
            df = pd.concat([df,lastdata],axis=0)
        else:
            df = download_symbol_data(symbol)
    except:
        # df = download_symbol_data(symbol)
        print("ERROR | update_data_symbol ")
    
    
    return save_data_pickle(df, symbol)


#update features for symbol
def update_data_features_symbol(symbol):
    df = pd.DataFrame()
    # update data file and get it
    if(update_data_symbol(symbol)):
        df_data = get_data(symbol)

        if(isinstance(df_data, pd.DataFrame)):
            df_features = get_data_features(symbol)
            #nouveaux df_data des feature a rajouter
            if(isinstance(df_features, pd.DataFrame)):
                try:
                    last_date = df_features.index[-NBR_VALUE_FEATURE]
                except Exception as e:
                    last_date = df_features.index[0]

                df_data2 = df_data.loc[df_data.index > last_date]
                
                #nouveaux indi
                df_data2 = indi.get_all_indicators(df_data2)

                # concatenation des anciennes et nouvelles données
                df = pd.concat([df_features,df_data2],axis=0)

                save_data_features_pickle(df,symbol)
            else:
                df = get_data_features(symbol)
            
            return df
    


#update data from binance for symbols list
def update_data(symbols = get_symbol()):
    for symbol in symbols:
            update_data_symbol(symbol)

#try to save df in PATH_DATA and delete duplicate value, return false if df is useless
def save_data_pickle(df, symbol):
    if isinstance(df, pd.DataFrame):
        if(date.today() - timedelta(days=2) <= df.index[-1] and df.shape[0] > NBR_VALUE_FEATURE):
            try:
                df = df[~df.index.duplicated(keep='last')]
                df.to_pickle(f'{PATH_DATA}/{symbol}.pickle')
                if(DEBUG_DL):
                    print(f"Enregistrement data pour {symbol} d'une taille {df.shape}")
                save_symbol(symbol)
                return True
            except:
                print(f"ERROR {symbol} ouverture fichier data")
                return False
        else:
            save_bad_symbol(symbol)
            print(f"Crypto Morte Date dépassée {symbol} derniere date : {df.index[-1]} taille : {df.shape}")
            return False

#try to save df in PATH_FEATURE and delete duplicate value     
def save_data_features_pickle(df, symbol):
    try:
        df = df[~df.index.duplicated(keep='last')]
        thresh = int(df.shape[0] * 0.99)
        df = df.dropna(axis=1, thresh=thresh)
        df.to_pickle(f'{PATH_FEATURE}/{symbol}.pickle')
        if(DEBUG_DL):
            print(f"Enregistrement feature pour {symbol} d'une taille {df.shape}")
            return get_data_features(symbol)
        return True
    except:
        print(f"ERROR {symbol} ouverture fichier feature")
        return False

#get data from database if you have it, or download and save it
def get_data(symbol):
    try:
        df = pd.read_pickle(f'{PATH_DATA}/{symbol}.pickle').astype(float)
        return df
    except:
        try:
            df = download_symbol_data(symbol)
            if(save_data_pickle(df,symbol)):
                return df
            else:
                return False
        except:
            print(f"Impossible de recuprer les données {symbol}")
            return False

#get data from database if you have it, or try to create it and save it
def get_data_features(symbol):
    try:
        df = pd.read_pickle(f'{PATH_FEATURE}/{symbol}.pickle')
        try:
            df = df.astype(float)
        except:
            print(f"Impossible de convertir en float {symbol}")
            
        return df
    except:
        try:
            df = get_data(symbol)
            df = indi.get_all_indicators(df)
            save_data_features_pickle(df,symbol)
        except:
            print(f"Impossible de créer features pour : {symbol}")

    return False


def testImport():
    print("test")