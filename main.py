from shower_time import is_shower_now
from data_acquisition import acquire_data_from_wilga

from time import sleep

def run():
    data = acquire_data_from_wilga(30,False)
    predict_shower = is_shower_now(data)

if __name__ == "__main__":
    while True:
        run()
        sleep(60)