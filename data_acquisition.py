import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from passwords_gitignore import SecretPass


url="10.45.98.1:8086"
bucket = "wilga-prod"

org = "<my-org>"
token = SecretPass.org

print(token)