from comfort_temp import rooms_thermal_comfort
from energy_wasted import open_window_heater_on, calculating_energy_waste_score, no_people_watching_tv_on, fridge_on_door_open, score_for_current_hour_energy

from data_acquisition import acquire_data_from_wilga

from datetime import datetime


def do_calculating(data, best_shower_time, total_energy_sum, rooms = ["bathroom", "smallroom", "largeroom"]):
    temperature_score = rooms_thermal_comfort(data, rooms)["average"]
    windows_heater = open_window_heater_on(data, rooms)
    energy_waste_score = calculating_energy_waste_score(temperature_score, 
                            no_people_watching_tv_on(data),
                            fridge_on_door_open(data),
                            windows_heater["percentage"])
    window_alert = alert_window_open_heater_on(windows_heater["rooms_with_waste"])
    shower_hour_score = score_for_current_hour_energy(best_shower_time, datetime.today())
    daily_energy_score = find_daily_energy_score(total_energy_sum)
    return {
        "temperature_score" : round(temperature_score),
        "energy_waste_score" : energy_waste_score,
        "window_alert" : window_alert,
        "shower_time_score" : shower_hour_score
    }


def alert_window_open_heater_on(data_windows_heater):
    if len(data_windows_heater) > 0:
        return 1
    return 0


def find_daily_energy_score(energy):
    if datetime.now().month in [11,12,1,2,3]:
        if energy < 4:
            return round(energy / 4.2 * 3)
        return 4
    else:
        if energy < 2.5:
            return round(energy / 2.8 * 3)
        return 4