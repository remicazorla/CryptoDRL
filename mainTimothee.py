import pandas as pd
from lib.download import *
import time
import multiprocessing
import lib.timerLib as timeL

if __name__ == '__main__': 
    start = time.time()
    # update_data_features(symbols=get_symbol())
    print(f'MAIN time for process : {(time.time()-start):.6f}')
    print(get_data('AMPUSDT'))
    print(get_data_features('AMPUSDT'))