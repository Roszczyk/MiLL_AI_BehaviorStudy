import pandas as pd
from pathlib import Path
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta, timezone
import sys
import os

# to change when handle relative import
from IoT_System.data_acquisition import sort_measurements, sort_rooms, AcquiredData, delete_battery_info


def date_to_format(date):
    date = datetime.combine(date, datetime.min.time()).replace(hour=12)
    date = date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return date


def data_from_influx(date_from, url, bucket, org, token): 
    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )
    date_to = date_from + timedelta(days=1)
    date_from = date_to_format(date_from)
    date_to = date_to_format(date_to)
    QUERY = f'from(bucket:"{bucket}") \
        |> range(start: {date_from}, stop: {date_to}) \
        |> filter(fn:(r) => r._field == "value")'
    
    query_api = client.query_api()
    data_acquired = query_api.query(query=QUERY, org=org)

    data = []
    if len(data_acquired) > 0:
        for table in data_acquired:
            for record in table.records:
                if hasattr(record, "get_time") and hasattr(record, "get_value") and hasattr(record, "get_measurement") and hasattr(record, "values") and "entity_id" in record.values:
                    data.append(AcquiredData(record.get_time(), record.values["entity_id"], record.get_value(), record.get_measurement()))
    return data


def fill_none_with_mean(data_dict):
    non_none_values = [v for v in data_dict.values() if v is not None]
    if non_none_values:
        mean_value = sum(non_none_values) / len(non_none_values)
        return {key: (mean_value if value is None else value) for key, value in data_dict.items()}
    return data_dict


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
    result_dict = fill_none_with_mean(result_dict)
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
    result_dict = fill_none_with_mean(result_dict)
    return result_dict


def calculate_room_presence_percentage(data, room, date):
    stop_date = date + timedelta(days=1)
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
        sum_time = time_1 + time_0
        return time_1 / sum_time
    return None


def calculate_presence_percentage_for_rooms(data, rooms, stop_date):
    data_motion = sort_measurements(data, "motion")
    result_dict = dict()
    for room in rooms:
        result_dict.update({ room : calculate_room_presence_percentage(data_motion, room, stop_date) })
    result_dict = fill_none_with_mean(result_dict)
    return result_dict


def load_given_data():
    data_file = Path(__file__).parent.parent / "data_about_guests.xlsx"
    df = pd.read_excel(data_file)
    df["dzieci"] = df["dzieci"].fillna(0)
    df["liczba_osob"] = df["dzieci"] + df["os. dor "]
    df["od"] = pd.to_datetime(df["od"])
    df["do"] = pd.to_datetime(df["do"])
    return df


def prepare_one_df(given_data):
    begin_dict = {
        "people_count" : [],
        "month" : [],
        "largeroom_temp" : [],
        "smallroom_temp" : [],
        "bathroom_temp": [],
        "largeroom_hum": [],
        "smallroom_hum": [],
        "bathroom_hum": [],
        "largeroom_presence": [],
        "smallroom_presence": [],
        "bathroom_presence": []
    }
    df = pd.DataFrame(begin_dict)
    date = None
    for index, row in given_data.iterrows():
        if date!=None:
            while date != row["od"].replace(hour = 12, tzinfo=timezone.utc):
                new_df_row = dict()
                data = data_from_influx(date, "10.45.98.1:8086", "wilga-prod", "confidential", "confidential")
                data = delete_battery_info(data)
                new_df_row.update({ "people_count" : 0 })
                new_df_row.update({ "month" : date.month })
                rooms_temp = calculate_average_rooms_temperatures(data, ["largeroom", "smallroom", "bathroom"])
                new_df_row.update({ "largeroom_temp" : rooms_temp["largeroom"] })
                new_df_row.update({ "smallroom_temp" : rooms_temp["smallroom"] })
                new_df_row.update({ "bathroom_temp" : rooms_temp["bathroom"] })
                rooms_hum = calculate_average_rooms_humidities(data, ["largeroom", "smallroom", "bathroom"])
                new_df_row.update({ "largeroom_hum" : rooms_hum["largeroom"] })
                new_df_row.update({ "smallroom_hum" : rooms_hum["smallroom"] })
                new_df_row.update({ "bathroom_hum" : rooms_hum["bathroom"] })
                presence_percentage = calculate_presence_percentage_for_rooms(data, ["largeroom", "smallroom", "bathroom"], date)
                new_df_row.update({ "largeroom_presence" : presence_percentage["largeroom"] })
                new_df_row.update({ "smallroom_presence" : presence_percentage["smallroom"] })
                new_df_row.update({ "bathroom_presence" : presence_percentage["bathroom"] })
                if len([v for v in new_df_row.values() if v is None])==0:
                    df = df.append(new_df_row, ignore_index=True)
                date = date + timedelta(days=1)
        else:
            date = row["od"]
        date = date.replace(hour = 12, tzinfo=timezone.utc)
        while date != row["do"].replace(hour = 12, tzinfo=timezone.utc):
            new_df_row = dict()
            data = data_from_influx(date, "10.45.98.1:8086", "wilga-prod", "confidential", "confidential")
            data = delete_battery_info(data)
            new_df_row.update({ "people_count" : row["liczba_osob"]})
            new_df_row.update({ "month" : date.month })
            rooms_temp = calculate_average_rooms_temperatures(data, ["largeroom", "smallroom", "bathroom"])
            new_df_row.update({ "largeroom_temp" : rooms_temp["largeroom"] })
            new_df_row.update({ "smallroom_temp" : rooms_temp["smallroom"] })
            new_df_row.update({ "bathroom_temp" : rooms_temp["bathroom"] })
            rooms_hum = calculate_average_rooms_humidities(data, ["largeroom", "smallroom", "bathroom"])
            new_df_row.update({ "largeroom_hum" : rooms_hum["largeroom"] })
            new_df_row.update({ "smallroom_hum" : rooms_hum["smallroom"] })
            new_df_row.update({ "bathroom_hum" : rooms_hum["bathroom"] })
            presence_percentage = calculate_presence_percentage_for_rooms(data, ["largeroom", "smallroom", "bathroom"], date)
            new_df_row.update({ "largeroom_presence" : presence_percentage["largeroom"] })
            new_df_row.update({ "smallroom_presence" : presence_percentage["smallroom"] })
            new_df_row.update({ "bathroom_presence" : presence_percentage["bathroom"] })
            if len([v for v in new_df_row.values() if v is None])==0:
                df = df.append(new_df_row, ignore_index=True)
            date = date + timedelta(days=1)
    return df


if __name__ == "__main__":
    given_data = load_given_data()
    df = prepare_one_df(given_data)
    print(df)
    df.to_csv(Path(__file__).parent / "prepared_data.csv")