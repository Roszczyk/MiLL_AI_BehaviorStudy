import pandas as pd
from pathlib import Path
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta, timezone
import sys
import os

# to change when handle relative import
from data_acquisition import sort_measurements, sort_rooms, AcquiredData, delete_battery_info


def date_to_format(date):
    date = datetime.combine(date, datetime.min.time()).replace(hour=12)
    date = date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return date


def data_from_influx(date_from, date_to, url, bucket, org, token): 
    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )
    date_from = date_to_format(date_from)
    date_to = date_to_format(date_to)
    QUERY = f'from(bucket:"{bucket}") \
        |> range(start: {date_from}, stop: {date_to}) \
        |> filter(fn:(r) => r._field == "value")'
    
    query_api = client.query_api()
    data_acquired = query_api.query(query=QUERY, org=org)

    data = []
    for table in data_acquired:
        for record in table.records:
            data.append(AcquiredData(record.get_time(), record.values["entity_id"], record.get_value(), record.get_measurement()))
    return data


def calculate_average_room_temperature(data, room):
    room_data = sort_rooms(data, room)
    if len(room_data) > 0:
        temperature_sum = 0
        for row in room_data:
            temperature_sum = temperature_sum + row.value
        return temperature_sum/len(room_data)
    return None


def calculate_average_rooms_temperatures(data, rooms):
    data_temp = sort_measurements(data, "temperature")
    result_dict = dict()
    for room in rooms:
        result_dict.update({ room : calculate_average_room_temperature(data_temp, room) })
    return result_dict


def calculate_average_room_humidity(data, room):
    room_data = sort_rooms(data, room)
    if len(room_data) > 0:
        humidity_sum = 0
        for row in room_data:
            humidity_sum = humidity_sum + row.value
        return humidity_sum/len(room_data)
    return None


def calculate_average_rooms_humidities(data, rooms):
    data_temp = sort_measurements(data, "humidity")
    result_dict = dict()
    for room in rooms:
        result_dict.update({ room : calculate_average_room_humidity(data_temp, room) })
    return result_dict


def calculate_room_presence_percentage(data, room, stop_date):
    room_data = sort_rooms(data, room)
    if len(room_data) == 1:
        return room_data[0].value
    if len(room_data) > 1:
        time_1 = 0
        time_0 = 0
        for i in range(len(room_data)-1):
            time_diff = room_data[i+1].time - room_data[i].time
            time_diff = time_diff.total_seconds()
            if room_data[i].value == 1:
                time_1 = time_1 + time_diff
            elif room_data[i].value == 0:
                time_0 = time_0 + time_diff
        time_diff = stop_date - room_data[len(room_data)-1].time
        time_diff = time_diff.total_seconds()
        if room_data[len(room_data)-1].value == 1:
            time_1 = time_1 + time_diff
        elif room_data[len(room_data)-1].value == 0:
            time_0 = time_0 + time_diff
        print(time_1, time_0)
        sum_time = time_1 + time_0
        return time_1 / sum_time
    return None


def calculate_presence_percentage_for_rooms(data, rooms, stop_date):
    data_motion = sort_measurements(data, "motion")
    result_dict = dict()
    for room in rooms:
        result_dict.update({ room : calculate_room_presence_percentage(data_motion, room, stop_date) })
    return result_dict


def load_given_data():
    data_file = Path(__file__).parent / "raw_data.xlsx"
    df = pd.read_excel(data_file)
    df["dzieci"] = df["dzieci"].fillna(0)
    df["liczba_osob"] = df["dzieci"] + df["os. dor "]
    df["od"] = pd.to_datetime(df["od"])
    df["do"] = pd.to_datetime(df["do"])
    return df

data = CONFIDENTIAL
rooms_temp = calculate_average_rooms_temperatures(data, ["largeroom", "smallroom", "bathroom"])
rooms_hum = calculate_average_rooms_humidities(data, ["largeroom", "smallroom", "bathroom"])
date_stop = datetime.today().replace(month=10, day=5).date()
date_stop = datetime.combine(date_stop, datetime.min.time()).replace(hour=12,tzinfo=timezone.utc)
presence_percentage = calculate_presence_percentage_for_rooms(data, ["largeroom", "smallroom", "bathroom"], date_stop)
print(rooms_temp, presence_percentage, rooms_hum)