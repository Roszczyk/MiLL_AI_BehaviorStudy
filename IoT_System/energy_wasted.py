from data_acquisition import acquire_data_from_wilga, sort_anything, sort_rooms, sort_measurements

from datetime import timedelta, datetime, timezone


def current_energy_sum(data, iteration = 1):
    if iteration > 15:
        print("SYSTEM ERROR - NO ENERGY DATA")
        return -1
    energy_data = sort_measurements(sort_anything(data, "total"), "energy")
    if len(energy_data) > 0:
        return energy_data[-1].value
    else:
        return current_energy_sum(acquire_data_from_wilga(900 + iteration*900), iteration=iteration+1)
    

def lights_on_no_presence(data, is_any):
    lights_data = sort_measurements(sort_anything(data, "lights"),"power")
    if len(lights_data) == 0:
        return False
    if lights_data[-1].value > 0 and not is_any:
        return True
    return False


def check_room_waste_open_window_heater(data_power, data_open_windows):
    if len(data_power) > 0 and len(data_open_windows) > 0: 
        any_window_open = False
        windows_types = []
        i = 1
        while i != len(data_open_windows) and not any_window_open:
            if data_open_windows[-i].entity not in windows_types:
                windows_types.append(data_open_windows[-i].entity)
                if data_open_windows[-i].value == 1.0:
                    any_window_open = True
            i = i + 1
        if (datetime.now(timezone.utc) - data_power[-1].time) > timedelta(minutes=8) \
                and data_power[-1].value > 0 and any_window_open:
            return True
    return False


def get_heater_power(data):
    return sort_measurements(sort_anything(data, "heater"), "power")


def open_window_heater_on(data, data_power, rooms):
    data_open_windows = sort_anything(data, "window")
    rooms_with_waste = []
    for room in rooms:
        if check_room_waste_open_window_heater(sort_rooms(data_power,room), sort_rooms(data_open_windows, room)):
            rooms_with_waste.append(room)
    return {
        "rooms_with_waste" : rooms_with_waste,
        "percentage" : len(rooms_with_waste)/len(rooms)
    }


def no_people_watching_tv_on(data):
    data_tv_on = sort_measurements(sort_anything(data, "tv"), "power")
    data_presence = sort_measurements(sort_rooms(data, "largeroom"), "motion")
    if len(data_tv_on)>0 and len(data_presence)>0:
        if (datetime.now(timezone.utc) - data_presence[-1].time) > timedelta(minutes=10) \
                and ((data_tv_on[-1].time - data_presence[-1].time) > timedelta(minutes=5) \
                     or (data_tv_on[-1].time - data_presence[-1].time) < timedelta(minutes=5)) \
                and data_presence[-1].value == 1.0 \
                and data_tv_on[-1].value > 0:
            return True
    return False


def fridge_on_door_open(data):
    data_fridge_on = sort_measurements(sort_anything(data, "fridge"), "power")
    data_fridge_door = sort_measurements(sort_anything(data, "fridge"), "door")
    if len(data_fridge_door) > 0 and len(data_fridge_door):
        if (datetime.now(timezone.utc) - data_fridge_on[-1].time) < timedelta(minutes=10) \
                and data_fridge_door[-1].value == 1.0 \
                and data_fridge_on[-1].value > 0:
            return True
    return False


def translate_expected_temperature_compare_to_bool(value):
    if value<=2 or value==None:
        return False
    else:
        return True
        

def calculating_energy_waste_score(temperatures, tv, fridge, windows, lights, heater_power):
    score = 0
    if len(heater_power) == 0 or heater_power[-1].value <= 0:
        temperatures = 0
    else:
        temperatures = max(0, temperatures-2)
    score = score + temperatures + 2*windows
    if tv:
        score = score + 0.5
    if fridge:
        score = score + 1.5
    if lights:
        score = score + 1.0
    score = max(0, min(score/3.5*4, 4))
    return round(score)


def score_for_current_hour_energy(best_hour, current_hour):
    delta = best_hour - current_hour
    if delta > timedelta(hours=1, minutes=30):
        return 0 # najlepsza godzina na wykorzystanie energii jeszcze się nie zbliża
    if delta <= timedelta(hours=1, minutes=30) and delta > timedelta(hours=0, minutes=15):
        return 1 # zbliża się najlepsza godzina
    if delta <= timedelta(minutes=15) and delta > timedelta(hours=-1):
        return 2 # najlepsza godzina na wykorzystanie energii
    if delta <= timedelta(hours=-1) and delta > timedelta(hours=-2):
        return 3 # najlepsza godzina na wykorzystanie energii właśnie minęła
    return 4 #tego dnia najlepsza godzina już minęła

