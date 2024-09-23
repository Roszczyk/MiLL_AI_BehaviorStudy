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
            print(rooms_with_waste)
        else:
            print(room, "all good :)")
    return rooms_with_waste

open_window_heater_on(acquire_data_from_wilga(900), ["bathroom", "smallroom", "largeroom"])