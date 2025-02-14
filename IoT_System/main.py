from shower_time import shower_handler, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from interface_joint import do_calculating
from mqtt_publisher import init_mqtt_publisher_for_wilga
from presence import is_someone_present
from energy_wasted import current_energy_sum

from time import sleep, time
from datetime import datetime
import os
import sys
from platform import processor
from threading import Thread


class StateOfObject:
    def __init__(self, rooms, friendly_name, sleep_time = 300, acquire_time = 900):
        self.friendly_name = friendly_name
        self.current_date = None
        self.rooms = rooms
        self.is_shower_now = False
        self.shower_prediction = None
        self.previous_energy_sum = 0
        self.is_someone = False
        self.sleep_time = sleep_time
        self.acquire_time = acquire_time

    def put_current_date(self):
        self.current_date = datetime.today().date()

    def reset_daily_energy_sum(self, current_energy):
        self.previous_energy_sum = current_energy

    def get_daily_energy_sum(self, current_energy):
        return current_energy - self.previous_energy_sum

    def run(self, mqtt):
        data = acquire_data_from_wilga(self.acquire_time)
        current_energy_status = current_energy_sum(data)
        if self.current_date != datetime.today().date():
            self.put_current_date()
            self.reset_daily_energy_sum(current_energy_status)
            self.shower_prediction = None
        today_energy_sum = self.get_daily_energy_sum(current_energy_status)
        detect_shower = shower_handler(data, self.is_shower_now)
        self.is_shower_now = detect_shower
        presense_info = is_someone_present(data, self.rooms)
        self.is_someone = presense_info["result"]
        if self.shower_prediction == None or self.is_shower_now == True:
            self.shower_prediction = calculate_hour_for_shower(datetime.today().replace(hour=17, minute=0, second=0))
        score = do_calculating(data, self.shower_prediction, today_energy_sum, self.is_someone, rooms = self.rooms)
        energy_waste_score = score["energy_waste_score"]
        temperature_score = score["temperature_score"]
        shower_time_score = score["shower_time_score"]
        window_alert = score["window_alert"]
        daily_energy_score = score["daily_energy_score"]
        mqtt.publish_for_interface_joint(score, self.friendly_name)
        print(f"\nDAY:{self.current_date}\n\nSCORES:\nenergy waste: {energy_waste_score}\ntemperature: {temperature_score}\nshower time: {shower_time_score}\
            \nwindow alert: {window_alert}\nenergy score: {daily_energy_score} ({today_energy_sum}, {current_energy_status})\n\nDETECTIONS:\nshower: {detect_shower}\npresense: {self.is_someone}\n")

        file = open(f"data_collection/{self.friendly_name}/iterations_logs.csv", "a")
        file.write(f"{datetime.now()},{energy_waste_score},{temperature_score},{shower_time_score},{window_alert},{daily_energy_score},{detect_shower},{self.is_someone},{today_energy_sum}\n")
        file.close()

    def manage_files(self):
        os.makedirs(f"data_collection/{self.friendly_name}", exist_ok=True)
        if not os.path.exists(f"data_collection/{self.friendly_name}/iterations_logs.csv"):    
            file = open(f"data_collection/{self.friendly_name}/iterations_logs.csv", "a")
            file.write("time, energy_waste, temperature, shower_time, window_alert, daily_energy_score, is_shower, is_present, energy_today\n")
            file.close()

    def thread_run(self, mqtt, sleep_time=None):
        if sleep_time == None:
            sleep_time = self.sleep_time
        performance_count_times = []
        while True:
            begin = time()
            self.run(mqtt)
            time_of_loop = time()-begin
            performance_count_times = save_performance(time_of_loop, performance_count_times, 10)
            print("time of loop: ", time_of_loop, "\n")
            sleep(sleep_time)


def save_performance(time_of_loop, times, thread_name, avg_out_of=10):
    times.append(time_of_loop)
    if len(times) >= avg_out_of:
        with open("data_collection/performance.txt", "a") as f:
            f.write(f"{thread_name}: {sum(times)/len(times)}, {processor()}\n")
        times = []
    return times


if __name__ == "__main__":
    # declare managed houses:
    houses_managed = [
        StateOfObject(["bathroom", "largeroom", "smallroom"], "56", 60) # house 56
    ]
    os.makedirs("data_collection", exist_ok=True)
    mqtt = init_mqtt_publisher_for_wilga()
    thread_list = []
    for h in houses_managed:
        h.manage_files()
        thread_list.append(Thread(target = h.thread_run, args=(mqtt, h.sleep_time), daemon=True))
    for thread in thread_list:
        thread.start()
    # enabling quitting the program with Ctrl+C:
    try:
        while True:
            sleep(0.1)
    except KeyboardInterrupt:
        sys.exit(1)