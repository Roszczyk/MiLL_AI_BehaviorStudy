import pandas as pd
from pathlib import Path
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
import sys
import os


# to delete when handle relative import
class AcquiredData:
    def __init__(self, time, entity, value, unit):
        self.time = time
        self.entity = entity
        self.value = value
        self.unit = unit


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


def load_given_data():
    data_file = Path(__file__).parent / "raw_data.xlsx"
    df = pd.read_excel(data_file)
    df["dzieci"] = df["dzieci"].fillna(0)
    df["liczba_osob"] = df["dzieci"] + df["os. dor "]
    df["od"] = pd.to_datetime(df["od"])
    df["do"] = pd.to_datetime(df["do"])
    return df

print(data_from_influx(datetime.today().replace(month=9, day=15).date(), datetime.today().replace(month=9, day=16).date(), "10.45.98.1:8086", "wilga-prod", "a896b376fd44040b", "s2Si7D6sxRmCo0ccP1Ua5IPeywU5AisGHmIlqMt7iRYQRA7GYJUONslENSEqaNxsPluGPg6cDaLOEXYTbYwZsg=="))
