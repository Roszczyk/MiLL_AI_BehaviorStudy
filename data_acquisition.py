import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from passwords_gitignore import get_token, get_org

URL="10.45.98.1:8086"
BUCKET = "wilga-prod"
ORG = get_org()
TOKEN = get_token()

client = influxdb_client.InfluxDBClient(
   url=URL,
   token=TOKEN,
   org=ORG
)

query_api = client.query_api()

QUERY = f'from(bucket:"{BUCKET}") \
    |> range(start: -10m) \
    |> filter(fn:(r) => r._field == "value")'

data_acquired = query_api.query(query=QUERY, org=ORG)

for table in data_acquired:
    for record in table.records:
        print(record.get_measurement(), record.get_value())