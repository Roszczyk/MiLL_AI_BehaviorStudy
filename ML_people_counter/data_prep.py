import pandas as pd
from pathlib import Path
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import sys
import os

# to change when handle relative import
from data_acquisition import sort_measurements, sort_rooms, AcquiredData


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
        temperatures = []
        for row in room_data:
            temperatures.append(row.value)
        return sum(temperatures)/len(temperatures)
    return None


def calculate_average_rooms_temperatures(data, rooms):
    data_temp = sort_measurements(data, "temperature")
    result_dict = dict()
    for room in rooms:
        result_dict.update({ room : calculate_average_room_temperature(data_temp, room) }) 
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
print(rooms_temp)