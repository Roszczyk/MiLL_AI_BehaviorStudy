import requests
from datetime import datetime, timedelta
import json

def get_energy_cost_predictions(start_time, end_time):
    params = {
        "date_from" : start_time.strftime('%d-%m-%YT%H:%M:%SZ'),
        "date_to" : end_time.strftime('%d-%m-%YT%H:%M:%SZ')
    }
    url = "https://energy-instrat-api.azurewebsites.net/api/prices/energy_price_rdn_hourly"
    response = requests.get(url, params=params)

    return json.loads(response.text)


def postprocess_data(raw_data):
    post_data = []
    for row in raw_data:
        post_data.append({
            "date" : row["date"],
            "cost" : row["fixing_ii"]["price"]
        })
    return post_data


def find_best_hour_energy(data):
    current_best = {"date" : None, "cost" : float("inf")}
    for row in data:
        if row["cost"] < current_best["cost"]:
            current_best = row
    return current_best


def do_find_best_hour_energy(start, end):
    raw_data = get_energy_cost_predictions(start, end)
    postprocessed = postprocess_data(raw_data)
    best_hour = find_best_hour_energy(postprocessed)["date"]
    best_hour = datetime.strptime(best_hour, '%Y-%m-%dT%H:%M:%SZ')
    return best_hour


if __name__ == "__main__":
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    print(do_find_best_hour_energy(today, tomorrow))