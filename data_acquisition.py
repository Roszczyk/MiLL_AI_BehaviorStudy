import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from passwords_gitignore import get_token, get_org

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

    print(data_acquired)

    for table in data_acquired:
        for record in table.records:
            print(record.get_time(), record.values["entity_id"], record.get_value(), record.get_measurement())

if __name__ == "__main__":
    URL="10.45.98.1:8086"
    BUCKET = "wilga-prod"
    ORG = get_org()
    TOKEN = get_token()

    acquire_data(URL, BUCKET, ORG, TOKEN, 10)