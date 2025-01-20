from data_acquisition import acquire_data_from_wilga, sort_rooms, sort_measurements
from dynamic_energy_cost import do_find_best_hour_energy
from datetime import datetime, timedelta, timezone


def count_average(data_array):
    sum = 0
    for row in data_array:
        sum = sum + row.value
    return sum / len(data_array)


def is_shower_now(data):
    data_bathroom = sort_rooms(data, "bathroom")
    data_sorted = sort_measurements(data_bathroom)
    data_motion = data_sorted["presence"]
    data_humidity = data_sorted["humidity"]
    data_temperature = data_sorted["temperature"]
    if len(data_motion)==0 or (datetime.now(timezone.utc) - data_motion[-1].time) > timedelta(minutes=5) \
                or data_motion[-1].value==0.0:
        return False
    if len(data_temperature) == 0 or len(data_humidity) == 0:
        return False
    if data_humidity[-1].value - count_average(data_humidity) > 4 and \
                data_temperature[-1] - count_average(data_temperature) > 10 and \
                (datetime.now(timezone.utc) - data_temperature[-1].time) < timedelta(minutes=5):
        return True
    else:
        return False
    

def shower_handler(data, previous):
    result = is_shower_now(data)
    if result and not previous:
        file = open("data_collection/showers_detected.txt", "a")
        file.write(f"{datetime.now(timezone.utc)}, ")
        file.close()
    if not result and previous:
        file = open("data_collection/showers_detected.txt", "a")
        file.write(f"{datetime.now(timezone.utc)}\n")
        file.close()
    return result


def calculate_hour_for_shower(expected_from_model):
    potential_start = expected_from_model - timedelta(hours=2)
    potential_end = expected_from_model + timedelta(hours=3)
    best_hour = do_find_best_hour_energy(potential_start, potential_end)
    return best_hour
    

if __name__ == "__main__":
    # print(score_for_current_hour(datetime.today().replace(hour=21, minute=30, second=0), datetime.today().replace(hour=21, minute=15, second=0)))
    pass