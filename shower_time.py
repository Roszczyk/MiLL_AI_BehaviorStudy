from data_acquisition import acquire_data_from_wilga, sort_rooms, sort_measurements
from datetime import datetime, timedelta, timezone

def is_shower_now(data):
    data_motion = sort_measurements(sort_rooms(data, "bathroom"), "motion")
    data_humidity = sort_measurements(sort_rooms(data, "bathroom"), "humidity")
    data_temperature = sort_measurements(sort_rooms(data, "bathroom"), "temperature")
    if len(data_motion)==0 or (datetime.now(timezone.utc) - data_motion[-1].time) > timedelta(minutes=5) \
                or data_motion[-1].value==0.0:
        return False
    

print(is_shower_now(acquire_data_from_wilga(180)))