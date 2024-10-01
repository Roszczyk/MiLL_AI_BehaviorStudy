from shower_time import is_shower_now
from data_acquisition import acquire_data_from_wilga
from energy_wasted import do_calculating

from time import sleep

def run():
    data = acquire_data_from_wilga(900)
    detect_shower = is_shower_now(data)
    score = do_calculating(data)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    print(energy_waste_score, temperature_score, detect_shower)

if __name__ == "__main__":
    while True:
        run()
        sleep(60)