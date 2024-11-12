from shower_time import shower_handler, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from interface_joint import do_calculating
from mqtt_publisher import MQTT_Publisher
from presence import is_someone_present
from energy_wasted import current_energy_sum

from time import sleep, time
from datetime import datetime


class StateOfObject:
    def __init__(self, rooms):
        self.current_date = None
        self.rooms = rooms
        self.is_shower_now = False
        self.previous_energy_sum = 0

    def put_current_date(self):
        self.current_date = datetime.today().date()

    def reset_daily_energy_sum(self, current_energy):
        self.previous_energy_sum = self.previous_energy_sum + current_energy

    def get_day_energy_sum(self, current_energy):
        return current_energy - self.previous_energy_sum


def run(state, mqtt):
    data = acquire_data_from_wilga(900)
    current_energy_status = current_energy_sum(data)
    if state.current_date != datetime.today().date():
        state.put_current_date()
        state.reset_daily_energy_sum(current_energy_status)
    today_energy_sum = state.get_day_energy_sum(current_energy_status)
    detect_shower = shower_handler(data, state.is_shower_now)
    state.is_shower_now = detect_shower
    best_shower_time = calculate_hour_for_shower(datetime.today().replace(hour=17, minute=0, second=0))
    score = do_calculating(data, best_shower_time, rooms = state.rooms)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    shower_time_score = score["shower_time_score"]
    window_alert = score["window_alert"]
    presense_info = is_someone_present(data, state.rooms)
    is_someone = presense_info["result"]

    # mqtt.publish_for_interface_joint(score)

    print(f"\nDAY:{state.current_date}\n\nSCORES:\nenergy waste: {energy_waste_score}\ntemperature: {temperature_score}\nshower time: {shower_time_score}\
          \nwindow alert: {window_alert}\n\nDETECTIONS:\nshower: {detect_shower}\npresense: {is_someone}\n")


if __name__ == "__main__":
    house_55 = StateOfObject(["bathroom", "largeroom", "smallroom"])
    # mqtt = MQTT_Publisher("username", "password", "broker_ip", "broker_port")
    mqtt = None
    while True:
        begin = time()
        run(house_55, mqtt)
        print("time of loop: ", time()-begin, "\n")
        sleep(2)