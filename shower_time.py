from data_acquisition import acquire_data_from_wilga, sort_rooms, sort_measurements
from dynamic_energy_cost import do_find_best_hour_energy
from datetime import datetime, timedelta, timezone


def count_average(data_array):
    sum = 0
    for row in data_array:
        sum = sum + row.value
    return sum / len(data_array)

def is_shower_now(data):
    data_motion = sort_measurements(sort_rooms(data, "bathroom"), "motion")
    data_humidity = sort_measurements(sort_rooms(data, "bathroom"), "humidity")
    data_temperature = sort_measurements(sort_rooms(data, "bathroom"), "temperature")
    if len(data_motion)==0 or (datetime.now(timezone.utc) - data_motion[-1].time) > timedelta(minutes=5) \
                or data_motion[-1].value==0.0:
        return False
    if data_humidity[-1].value - count_average(data_humidity) > 4 and \
                data_temperature[-1] - count_average(data_temperature) > 2 and \
                (datetime.now(timezone.utc) - data_temperature[-1].time) < timedelta(minutes=5):
        return True
    else:
        return False


def calculate_hour_for_shower(expected_from_model):
    potential_start = expected_from_model - timedelta(hours=2)
    potential_end = expected_from_model + timedelta(hours=3)
    best_hour = do_find_best_hour_energy(potential_start, potential_end)
    return best_hour
    

if __name__ == "__main__":
    # print(score_for_current_hour(datetime.today().replace(hour=21, minute=30, second=0), datetime.today().replace(hour=21, minute=15, second=0)))
    pass