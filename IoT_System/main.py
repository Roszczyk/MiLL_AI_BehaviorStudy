from shower_time import shower_handler, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from interface_joint import do_calculating
from mqtt_publisher import MQTT_Publisher
from presence import is_someone_present
from energy_wasted import current_energy_sum

from time import sleep, time
from datetime import datetime
import os
from platform import processor


class StateOfObject:
    def __init__(self, rooms, friendly_name):
        self.friendly_name = friendly_name
        self.current_date = None
        self.rooms = rooms
        self.is_shower_now = False
        self.previous_energy_sum = 0

    def put_current_date(self):
        self.current_date = datetime.today().date()

    def reset_daily_energy_sum(self, current_energy):
        self.previous_energy_sum = self.previous_energy_sum + current_energy

    def get_daily_energy_sum(self, current_energy):
        return current_energy - self.previous_energy_sum


def run(state, mqtt):
    data = acquire_data_from_wilga(900)
    current_energy_status = current_energy_sum(data)
    if state.current_date != datetime.today().date():
        state.put_current_date()
        state.reset_daily_energy_sum(current_energy_status)
    today_energy_sum = state.get_daily_energy_sum(current_energy_status)
    detect_shower = shower_handler(data, state.is_shower_now)
    state.is_shower_now = detect_shower
    best_shower_time = calculate_hour_for_shower(datetime.today().replace(hour=17, minute=0, second=0))
    score = do_calculating(data, best_shower_time, today_energy_sum, rooms = state.rooms)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    shower_time_score = score["shower_time_score"]
    window_alert = score["window_alert"]
    daily_energy_score = score["daily_energy_score"]
    presense_info = is_someone_present(data, state.rooms)
    is_someone = presense_info["result"]

    # mqtt.publish_for_interface_joint(score)
    print(f"\nDAY:{state.current_date}\n\nSCORES:\nenergy waste: {energy_waste_score}\ntemperature: {temperature_score}\nshower time: {shower_time_score}\
          \nwindow alert: {window_alert}\nenergy score: {daily_energy_score} ({today_energy_sum}, {current_energy_status})\n\nDETECTIONS:\nshower: {detect_shower}\npresense: {is_someone}\n")

    file = open("data_collection/iterations_logs.csv", "a")
    file.write(f"{datetime.now()},{energy_waste_score},{temperature_score}, {shower_time_score},{window_alert},{daily_energy_score},{detect_shower},{is_someone},{today_energy_sum}\n")
    file.close()


def save_performance(time_of_loop, times, avg_out_of=10):
    times.append(time_of_loop)
    if len(times) >= avg_out_of:
        with open("data_collection/performance.txt", "a") as f:
            f.write(f"{processor()}: {sum(times)/len(times)}")
        times = []
    return times


if __name__ == "__main__":
    house_56 = StateOfObject(["bathroom", "largeroom", "smallroom"], "56")
    os.makedirs("data_collection", exist_ok=True)
    performance_count_times = []
    # mqtt = MQTT_Publisher("username", "password", "broker_ip", "broker_port")
    mqtt = None
    if not os.path.exists("data_collection/iterations_logs.csv"):    
        file = open("data_collection/iterations_logs.csv", "a")
        file.write("time, energy_waste, temperature, shower_time, window_alert, daily_energy_score, is_shower, is_present, energy_today\n")
        file.close()
    while True:
        begin = time()
        run(house_56, mqtt)
        time_of_loop = time()-begin
        print("time of loop: ", time_of_loop, "\n")
        performance_count_times = save_performance(time_of_loop, performance_count_times, 10)
        sleep(60)