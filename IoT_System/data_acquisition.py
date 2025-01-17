import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from time import sleep
from datetime import datetime
import sys

from passwords_gitignore import get_token, get_org

class AcquiredData:
    def __init__(self, time, entity, value, unit):
        self.time = time
        self.entity = entity
        self.value = value
        self.unit = unit


def acquire_data(url, bucket, org, token, time_in_minutes=10):
    client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
    )

    query_api = client.query_api()

    QUERY = f'from(bucket:"{bucket}") \
        |> range(start: -{time_in_minutes}m) \
        |> filter(fn:(r) => r._field == "value")'

    data_acquired = query_api.query(query=QUERY, org=org)

    data = []

    for table in data_acquired:
        for record in table.records:
            data.append(AcquiredData(record.get_time(), record.values["entity_id"], record.get_value(), record.get_measurement()))

    return data


def ping(host):
    import subprocess
    try:
        cmd = ["ping", "-n", "1", host] if subprocess.os.name == "nt" else ["ping", "-c", "1", host]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        output = result.stdout.lower()
        if result.returncode == 1 or "unreachable" in output:
            return False
        return True
        
    except Exception as e:
        print(f"Błąd: {e}")
        sys.exit(1)
    

def handle_connection_error(time_in_minutes, battery_info, iteration, pings, URL, exception = "Network"):
    print(exception)
    print(f"trying to acquire data, remaining: {iteration} attempts, {pings} pings")
    with open("data_collection/error_logs.txt", "a") as f:
        f.write(f"time: {datetime.now()}\nexception: {exception}\niteration: {iteration}, pings: {pings}\n\n")
    if iteration==0:
        host_ip = URL.split(":")[0]
        is_reachable = ping(host_ip)
        if not is_reachable:
            print(f"Host {host_ip} is not reachable")
            sys.exit(1)
        else:
            if pings == 0:
                print(f"Connection error, host {host_ip} reachable")               
                sys.exit(1)
            return acquire_data_from_wilga(time_in_minutes, battery_info=battery_info, iteration = 5, pings = pings-1)
    sleep(60)
    return acquire_data_from_wilga(time_in_minutes, battery_info=battery_info, iteration = iteration-1, pings = pings)


def acquire_data_from_wilga(time_in_minutes = 10, battery_info = False, iteration = 10, pings = 2):
    URL="http://10.45.98.1:8086"
    BUCKET = "wilga-prod"
    ORG = get_org()
    TOKEN = get_token()

    try:
        data = acquire_data(URL, BUCKET, ORG, TOKEN, time_in_minutes)
    except Exception as e:
        data = handle_connection_error(time_in_minutes, battery_info, iteration, pings, URL, exception=e)


    if not battery_info:
        data = delete_battery_info(data)

    return data


def sort_rooms(data, room = None, house_rooms = ["bathroom", "largeroom", "smallroom"]):
    if room == None or room == "other":
        house_rooms.append("outside")
        print(house_rooms)
        output_dict = dict()
        for r in house_rooms:
            output_dict.update({r : []})
        other_rooms_data = []

        for row in data:
            found_room = False
            for r in house_rooms:
                if r in row.entity and not found_room:
                    output_dict[r].append(row)
                    found_room = True
            if not found_room:
                other_rooms_data.append(row)

        if room == "other":
            return other_rooms_data
        
        output_dict.update({"other" : other_rooms_data})

        return output_dict
    
    else:
        selected_room_data = []
        if room in ["bathroom", "largeroom", "smallroom", "outside"]:
            for row in data:
                if room in row.entity:
                    selected_room_data.append(row)
            return selected_room_data
        else:
            print("Available rooms: bathroom, largeroom, smallroom, outside, other")
            return None


def sort_measurements(data, measurement = None):
    if measurement == None:
        temperature = []
        humidity = []
        pressure = []
        power = []
        energy = []
        is_open = []
        presence = []
        other = []

        for row in data:
            if "temperature" in row.entity:
                temperature.append(row)
            elif "humidity" in row.entity:
                humidity.append(row)
            elif "pressure" in row.entity:
                pressure.append(row)
            elif "power" in row.entity:
                power.append(row)
            elif "energy" in row.entity:
                energy.append(row)
            elif "window" in row.entity or "door" in row.entity:
                is_open.append(row)
            elif "motion" in row.entity:
                presence.append(row)
            else:
                other.append(row)

        return {
            "temperature" : temperature, 
            "humidity" : humidity, 
            "pressure" : pressure, 
            "power" : power, 
            "is_open" : is_open, 
            "presence" : presence, 
            "other" : other
        }
    
    else:
        selected_data = []
        for row in data:
            if measurement in row.entity:
                selected_data.append(row)
        return selected_data


def sort_anything(data, key):
    new_data = []
    for row in data:
        if key in row.entity:
            new_data.append(row)
    return new_data


def delete_battery_info(data):
    new_array = []
    for row in data:
        if "battery" in row.entity:
            pass
        else:
            new_array.append(row)
    return new_array


if __name__ == "__main__":
    data = acquire_data_from_wilga(300)
    data = delete_battery_info(data)
    sorted_rooms_data = sort_rooms(data)
    print(sorted_rooms_data.keys())
    bathroom_data = sorted_rooms_data["bathroom"]
    other_rooms_data = sorted_rooms_data["other"]