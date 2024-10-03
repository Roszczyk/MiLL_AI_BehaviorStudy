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
    SET = ptc.set_tmp(tdb = temperature, 
                  tr = temperature_radian, 
                  v = wind_velocity, 
                  rh = humidity, 
                  met = met, 
                  clo = clo)
    return SET


def get_PMV(temperature, temperature_radian, humidity, wind_velocity = 0, met = 1.2, clo = 1.0):
    # oczekiwane średnie odczucia osób w skali (-3,3), gdzie 0 to najlepszy komfort
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


def expected_thermal_comfort(average_temperature, room_temperature, room_humidity):
    PMV = get_PMV(room_temperature, average_temperature, room_humidity)
    if PMV >= -0.5 and PMV <= 0.5:
        return 2
    if PMV < -0.5 and PMV >= -1.1:
        return 1
    if PMV < -1.1:
        return 0
    if PMV > 0.5 and PMV <= 1.1:
        return 3
    if PMV > 1.1:
        return 4
    
    
def rooms_thermal_comfort(data, rooms, minutes = 900, iterations = 5):
    data_temperatures = sort_measurements(data, "temperature")
    data_humidities = sort_measurements(data, "humidity")
    list_of_temps = []
    dict_for_rooms = dict()
    for room in rooms:
        room_temperature = sort_rooms(data_temperatures, room)
        room_humidity = sort_rooms(data_humidities, room)
        dict_for_rooms.update({room : []})
        if len(room_temperature) != 0:
            room_temperature = room_temperature[-1].value
            dict_for_rooms[room].append(room_temperature)
            list_of_temps.append(room_temperature)
        else: 
            dict_for_rooms[room].append(None)
        if len(room_humidity) != 0:
            room_humidity = room_humidity[-1].value
            dict_for_rooms[room].append(room_humidity)
        else:
            dict_for_rooms[room].append(50)
    if len(list_of_temps) == 0:
        if iterations > 0:
            return rooms_thermal_comfort(acquire_data_from_wilga(minutes + 400), rooms, minutes + 400, iterations - 1)
        else:
            # RAPORT A PROBLEM
            average_temperature = 20.2
    else:
        average_temperature = sum(list_of_temps) / len(list_of_temps)
        dict_for_scores = dict()
        sum_for_average = 0
        for room in rooms:
            if dict_for_rooms[room][0] == None:
                dict_for_rooms[room][0] = average_temperature
            score = expected_thermal_comfort(average_temperature, dict_for_rooms[room][0], dict_for_rooms[room][0])
            dict_for_scores.update({room : score})
            sum_for_average = sum_for_average + score
        return {
            "room_scores" : dict_for_scores,
            "average" : sum_for_average/len(rooms)
        }



if __name__ == "__main__":
    pass