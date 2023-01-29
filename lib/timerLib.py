import time

#Seconds, Minutes, Hours
def get_time_from_ms(ms):
    seconds=(ms/1000)%60
    seconds = int(seconds)
    minutes=(ms/(1000*60))%60
    minutes = int(minutes)
    hours=(ms/(1000*60*60))%24
    hours = int(hours)

    return [seconds,minutes,hours]
    # return [int((ms/1000)%60),int((ms/(1000*60))%60),int((ms/(1000*60*60))%24) ]

def diff(ms):
    return (time.time()-ms) * 1000