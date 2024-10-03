from shower_time import is_shower_now, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from energy_wasted import do_calculating
from comfort_temp import get_outside_temperature_array, get_mean_outside_temperature

from time import sleep, time
from datetime import datetime

current_data = None
mean_outside_temperature = None

def run():
    global current_data, mean_outside_temperature
    if current_data != datetime.today().date() or mean_outside_temperature == None:
        print(current_data)
        current_data = datetime.today().date()
        print(current_data)
        mean_outside_temperature = get_mean_outside_temperature(get_outside_temperature_array())
    data = acquire_data_from_wilga(900)
    detect_shower = is_shower_now(data)
    best_shower_time = calculate_hour_for_shower(datetime.today().replace(hour=17, minute=0, second=0))
    score = do_calculating(data, best_shower_time)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    shower_time_score = score["shower_time_score"]

    print(energy_waste_score, temperature_score, detect_shower, shower_time_score)

if __name__ == "__main__":
    while True:
        begin = time()
        run()
        print("time of loop: ", time()-begin)
        sleep(2)