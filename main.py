from shower_time import is_shower_now
from data_acquisition import acquire_data_from_wilga
from energy_wasted import do_calculating
from comfort_temp import get_outside_temperature_array, get_mean_outside_temperature, get_comfort_indexes_from_data

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
    score = do_calculating(data)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    comfort_temperature_indexes = get_comfort_indexes_from_data(data, mean_outside_temperature)
    ASHRAE_index = comfort_temperature_indexes["ASHRAE"] #based on outside and inside temperatures
    PPD_index = comfort_temperature_indexes["PMV"] #number of people feeling comfortable in given conditions
    SET_index = comfort_temperature_indexes["SET"] #based on humidity, clothing, acitivity
    comfort_temperature = (ASHRAE_index + SET_index)/2

    print(comfort_temperature_indexes, comfort_temperature)

    print(energy_waste_score, temperature_score, detect_shower)

if __name__ == "__main__":
    while True:
        begin = time()
        run()
        print("time of loop: ", time()-begin)
        sleep(2)