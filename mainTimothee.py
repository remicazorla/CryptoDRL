import pandas as pd
from lib.download import *
import time
import multiprocessing
import lib.timerLib as timeL

if __name__ == '__main__': 
    start = time.time()

    update_data_features(symbols=get_pair_tickers())
    # update_data_features_symbol('BTCUSDT')
    timetab = timeL.get_time_from_ms(timeL.diff(start))
    print(f'MAIN time for process : {timetab[2]}h:{timetab[1]}m:{timetab[0]}s')

