import time

#Seconds, Minutes, Hours
def get_time_from_ms(ms):
    return [int((ms/1000)%60),int((ms/(1000*60))%60),int((ms/(1000*60*60))%24) ]

def diff(ms):
    return (time.time()-ms) * 1000

def get_time_needed(ms, nbrFinish,nbrTotal):
    return get_time_from_ms((diff(ms)/nbrFinish) * (nbrTotal - nbrFinish))