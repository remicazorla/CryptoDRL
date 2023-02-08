import lib.timerLib as timeL
import lib.indicators as indi
import lib.download as dl
import multiprocessing
from setup import *
import pandas as pd
import time

def update_all_db():
    dl.update_data_features(symbols=dl.get_symbol())

if __name__ == '__main__': 
    start = time.time()
    symbol = 'ETHUSDT'
    data = dl.get_data_features(symbol)
    print("data : ")
    print(data.shape)


    timetab = timeL.get_time_from_ms(timeL.diff(start))
    print(f'MAIN time for process : {timetab[2]}h:{timetab[1]}m:{timetab[0]}s')

