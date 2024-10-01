from data_acquisition import acquire_data_from_wilga, sort_anything, sort_rooms, sort_measurements

from datetime import timedelta, datetime, timezone


def check_room_waste(room_data):
    data_power = sort_measurements(sort_anything(room_data, "heater"), "power")
    data_open_windows = sort_anything(room_data, "window")
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


def open_window_heater_on(data, rooms):
    rooms_with_waste = []
    for room in rooms:
        if check_room_waste(sort_rooms(data,room)):
            rooms_with_waste.append(room)
    return {
        "rooms_with_waste" : rooms_with_waste,
        "percentage" : len(rooms_with_waste)/len(rooms)
    }


def no_people_watching_tv_on(data):
    data_tv_on = sort_measurements(sort_anything(data, "tv"), "power")
    data_presence = sort_measurements(sort_rooms(data, "largeroom"), "motion")
    if (datetime.now(timezone.utc) - data_presence[-1].time) > timedelta(minutes=10) \
            and data_presence[-1].value == 1.0 \
            and data_tv_on[-1].value > 0:
        return True
    return False


def fridge_on_door_open(data):
    data_fridge_on = sort_measurements(sort_anything(data, "fridge"), "power")
    data_fridge_door = sort_measurements(sort_anything(data, "fridge"), "door")
    if (datetime.now(timezone.utc) - data_fridge_on[-1].time) > timedelta(minutes=10) \
            and (datetime.now(timezone.utc) - data_fridge_door[-1].time) > timedelta(minutes=10) \
            and (data_fridge_door[-1].time - data_fridge_door[-2].time) > timedelta(minutes=2) \
            and data_fridge_door[-1].value == 1.0 \
            and data_fridge_on[-1].value > 0:
        return True
    return False


def expected_temperature_compare(data, expected_temperature, room):
    data_temperature = sort_measurements(sort_rooms(data, room), "temperature")
    if len(data_temperature)>0:
        if data_temperature[-1].value > expected_temperature * 1.1:
            return 4
        if data_temperature[-1].value > expected_temperature * 1.03:
            return 3
        if data_temperature[-1].value < expected_temperature * 0.9:
            return 0
        if data_temperature[-1].value < expected_temperature * 0.97:
            return 1
        else:
            return 2
    else:
        return 2
    

def translate_expected_temperature_compare_to_bool(value):
    if value<=2 or value==None:
        return False
    else:
        return True
    

def compare_rooms_temperature(data, rooms, expected_temperature=20, add_for_bathroom=2):
    dict_for_scores = dict()
    sum_for_average = 0
    for room in rooms:
        score = expected_temperature_compare(data, expected_temperature+add_for_bathroom, room)
        dict_for_scores.update({room : score})
        sum_for_average = sum_for_average + score
    return {
        "room_scores" : dict_for_scores,
        "average" : sum_for_average/len(rooms)
    }
        

def calculating_score(temperatures, tv, fridge, windows):
    score = 0
    temperatures = temperatures["average"]/4
    score = score + temperatures + windows["percentage"]
    if tv:
        score = score + 0.5
    if fridge:
        score = score + 0.5
    score = score/3*4
    return int(score)
    
if __name__ == "__main__":
    data = acquire_data_from_wilga(900)
    print(calculating_score(compare_rooms_temperature(data, ["bathroom", "smallroom", "largeroom"]), 
                            no_people_watching_tv_on(data),
                            fridge_on_door_open(data),
                            open_window_heater_on(data, ["bathroom", "smallroom", "largeroom"])))
