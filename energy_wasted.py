from data_acquisition import acquire_data_from_wilga, sort_anything, sort_rooms, sort_measurements

from datetime import timedelta

def check_room_waste(room_data):
    data_power = sort_measurements(sort_anything(room_data, "heater"), "power")
    data_open_windows = sort_anything(room_data, "window")
    if len(data_power) > 0 and len(data_open_windows) > 0:
        if abs(data_power[-1].time - data_open_windows[-1].time) < timedelta(minutes=5) \
                and data_power[-1].value > 0 and data_open_windows[-1].value == 1.0:
            return True
        print(data_power[-1].entity, data_open_windows[-1].entity)
    return False

def open_window_heater_on(data, rooms):
    rooms_with_waste = []
    for room in rooms:
        if check_room_waste(sort_rooms(data,room)):
            rooms_with_waste.append(room)
            print(rooms_with_waste)
        else:
            print(room, "all good :)")
    return rooms_with_waste

open_window_heater_on(acquire_data_from_wilga(350), ["bathroom", "smallroom", "largeroom"])