import pandas as pd
from lib.download import *
import time
import multiprocessing
import lib.timerLib as timeL


def update_all_db():
    update_data_features(symbols=get_symbol())

if __name__ == '__main__': 
    start = time.time()
    l = get_symbol()
    print("ok")
    time.sleep(5)
    save_symbol(l)
    timetab = timeL.get_time_from_ms(timeL.diff(start))
    print(f'MAIN time for process : {timetab[2]}h:{timetab[1]}m:{timetab[0]}s')

