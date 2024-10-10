from data_acquisition import sort_measurements, sort_rooms
from datetime import datetime, timedelta, timezone

def is_someone_present_in_room(data_motion, room):
    room_data = sort_rooms(data_motion, room)
    if len(room_data) == 0 or \
            (datetime.now(timezone.utc) - data_motion[-1].time) > timedelta(minutes=8) or \
            room_data[-1].value == 0.0:
        return False
    return True


def is_someone_present(data, rooms = ["bathroom", "largeroom", "smallroom"]):
    data_motion = sort_measurements(data, "motion")
    is_any = False
    rooms_presense = dict()
    for room in rooms:
        room_result = is_someone_present_in_room(data_motion, room)
        rooms_presense.update({room : room_result})
        if not is_any and room_result:
            is_any = True
    return {
        "rooms_presense" : rooms_presense,
        "result" : is_any
    }