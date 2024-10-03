import pythermalcomfort.models as ptc
import pythermalcomfort.utilities as utils

from meteostat import Point, Daily
from datetime import datetime, timedelta

from data_acquisition import sort_measurements, sort_rooms, acquire_data_from_wilga


def get_outside_temperature_array():
    location = Point(51.82908, 21.40699)
    end = datetime.today()
    start = end - timedelta(days=10)

    temperature_array = Daily(location, start, end)
    temperature_array = list(temperature_array.fetch()["tavg"])
    temperature_array.reverse()
    return temperature_array


def get_mean_outside_temperature(temp_array):
    # alpha (float) – constant between 0 and 1. The EN 16798-1 2019 3 recommends a value of 0.8, 
    # while the ASHRAE 55 2020 recommends to choose values between 0.9 and 0.6, corresponding to a 
    # slow- and fast- response running mean, respectively. Adaptive comfort theory suggests that a 
    # slow-response running mean (alpha = 0.9) could be more appropriate for climates in which 
    # synoptic-scale (day-to- day) temperature dynamics are relatively minor, such as the humid tropics.
    return utils.running_mean_outdoor_temperature(temp_array = temp_array, alpha = 0.8)


def get_SET(temperature, temperature_radian, humidity, wind_velocity = 0, met = 1.2, clo = 1.0):   
    # MET:
    # 0.8 – siedzenie, odpoczynek (np. praca biurowa, oglądanie telewizji).
    # 1.2 – lekkie czynności (np. praca w pozycji stojącej, praca przy komputerze).
    # 1.5 – spacer z prędkością 4 km/h (np. wolny marsz).
    # 2.0 – lekkie prace fizyczne (np. prace domowe, chodzenie po schodach).
    # 3.0 – szybki marsz, praca fizyczna (np. chodzenie z prędkością 6 km/h, intensywne sprzątanie).
    # 4.0 – bieganie z prędkością 8 km/h (np. trucht).
    # CLO:
    # 0.0 – brak ubrań lub bardzo lekkie ubrania (np. kąpielówki, strój plażowy).
    # 0.5 – lekkie ubrania (np. letnie ubrania, koszulka z krótkim rękawem, krótkie spodenki).
    # 1.0 – standardowe ubrania (np. długie spodnie, koszula, lekka marynarka lub sweter).
    # 1.5 – cięższe ubrania (np. zimowy płaszcz, ciepły sweter, szalik).
    # 2.0 – bardzo ciepłe ubrania (np. kurtka puchowa, grube spodnie, dodatkowe warstwy ubrań, odzież zimowa).
    # 3.0 – ekstremalne warunki (np. odzież na bardzo zimne warunki, kombinezon narciarski, odzież puchowa).
    SET = ptc.set_tmp(tdb = temperature, 
                  tr = temperature_radian, 
                  v = wind_velocity, 
                  rh = humidity, 
                  met = met, 
                  clo = clo)
    return SET


def get_PMV(temperature, temperature_radian, humidity, wind_velocity = 0, met = 1.2, clo = 1.0):
    # percentage of people feeling comfortable in the given conditions
    PMV = ptc.pmv(tdb = temperature, 
                  tr = temperature_radian, 
                  vr = wind_velocity, 
                  rh = humidity, 
                  met = met, 
                  clo = clo)
    return PMV


def get_ASHRAE(temperature, temperature_radian, mean_outside_temperature, wind_velocity = 0):
    ASHRAE = ptc.adaptive_ashrae(tdb = temperature, 
                tr = temperature_radian, 
                t_running_mean = mean_outside_temperature, 
                v = wind_velocity)
    return ASHRAE["tmp_cmf"]


def get_comfort_indexes_from_data(data, mean_outside_temperature, minutes = 900, iteration = 1):
    data_temperatures = sort_rooms(sort_measurements(data, "temperature"))
    data_humidity = sort_rooms(sort_measurements(data, "humidity"))
    count_temp = []
    count_hum = []
    bathroom_temp = None
    if len(data_temperatures["bathroom"]) > 0:
        bathroom_temp = data_temperatures["bathroom"][-1].value
    if len(data_temperatures["largeroom"]) > 0:
        count_temp.append(data_temperatures["largeroom"][-1].value)
        count_hum.append(data_humidity["largeroom"][-1].value)
    if len(data_temperatures["smallroom"]) > 0:
        count_temp.append(data_temperatures["smallroom"][-1].value)
        count_hum.append(data_humidity["smallroom"][-1].value)
    if len(count_temp) < 1:
        if iteration < 5:
            return get_comfort_indexes_from_data(acquire_data_from_wilga(minutes + 400), mean_outside_temperature, minutes + 400, iteration + 1)
        else:
            return{
                "ASHRAE" : 21,
                "PMV" : 0,
                "SET" : 21
            }
    average_temp = sum(count_temp)/len(count_temp)
    average_hum = sum(count_hum)/len(count_hum)
    if bathroom_temp != None:
        temperature_radian = (sum(count_temp) + bathroom_temp) / len(count_hum)
    else: 
        temperature_radian = average_temp

    SET = get_SET(average_temp, temperature_radian, average_hum)
    ASHRAE = get_ASHRAE(SET, temperature_radian, mean_outside_temperature)
    PMV = get_PMV(average_temp, temperature_radian, average_hum)

    return{
        "ASHRAE" : ASHRAE,
        "PMV" : PMV,
        "SET" : SET
    }



if __name__ == "__main__":
    # print(get_ASHRAE(22, get_mean_outside_temperature()), get_SET(22, 60), get_PMV(22,60))
    print(get_comfort_indexes_from_data(acquire_data_from_wilga(900),15))