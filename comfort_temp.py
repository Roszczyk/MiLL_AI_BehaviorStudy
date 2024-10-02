import pythermalcomfort.models as ptc
import pythermalcomfort.utilities as utils

from meteostat import Point, Daily
from datetime import datetime, timedelta

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

def get_temperature_array():
    location = Point(51.82908, 21.40699)
    end = datetime.today()
    start = end - timedelta(days=10)

    temperature_array = Daily(location, start, end)
    temperature_array = list(temperature_array.fetch()["tavg"])
    temperature_array.reverse()
    return temperature_array

mean_outside_temperature = utils.running_mean_outdoor_temperature(temp_array=get_temperature_array(), alpha=0.8)


def get_mean_outside_temperature():
    # alpha (float) – constant between 0 and 1. The EN 16798-1 2019 3 recommends a value of 0.8, 
    # while the ASHRAE 55 2020 recommends to choose values between 0.9 and 0.6, corresponding to a 
    # slow- and fast- response running mean, respectively. Adaptive comfort theory suggests that a 
    # slow-response running mean (alpha = 0.9) could be more appropriate for climates in which 
    # synoptic-scale (day-to- day) temperature dynamics are relatively minor, such as the humid tropics.
    return utils.running_mean_outdoor_temperature(temp_array=get_temperature_array(), alpha=0.8)


def get_adaptive_ASHRAE(temperature, mean_outside_temperature, wind_velocity = 0):
    ASHRAE = ptc.adaptive_ashrae(tdb = temperature, 
                tr = temperature, 
                t_running_mean = mean_outside_temperature, 
                v = wind_velocity)
    return ASHRAE["tmp_cmf"]


# print(ptc.set_tmp(tdb = 22,
#                 tr = 22,
#                 v = 0,
#                 rh = 60,
#                 met = 1.2,
#                 clo = 0.5))

print(get_adaptive_ASHRAE(22, get_mean_outside_temperature()))