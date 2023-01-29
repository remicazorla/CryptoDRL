import pandas as pd
from lib.download import *
import time
import multiprocessing
import lib.timerLib as timeL

if __name__ == '__main__': 
    start = time.time()
    symbols = get_pair_tickers()[:10]

    update_data_features(symbols=get_pair_tickers())
    # update_data_features_symbol('BTCUSDT')
    print(f'MAIN time for process : {(time.time()-start):.6f}')