from shower_time import shower_handler, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from interface_joint import do_calculating
from mqtt_publisher import MQTT_Publisher
from presence import is_someone_present

from time import sleep, time
from datetime import datetime
import os
from platform import processor


class StateOfObject:
    def __init__(self, rooms):
        self.current_energy_sum = 0
        self.current_date = None
        self.rooms = rooms
        self.is_shower_now = False

    def put_current_date(self):
        self.current_date = datetime.today().date()

    def reset_daily_energy_sum(self):
        self.current_energy_sum = 0


def run(state, mqtt):
    if state.current_date != datetime.today().date():
        state.put_current_date()
        state.reset_daily_energy_sum()
    data = acquire_data_from_wilga(900)
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

    file = open("data_collection/iterations_logs.csv", "a")
    file.write(f"{datetime.now()},{energy_waste_score},{temperature_score}, {shower_time_score},{window_alert},{detect_shower},{is_someone}\n")
    file.close()


def save_performance(time_of_loop, times, avg_out_of=10):
    times.append(time_of_loop)
    if len(times) >= avg_out_of:
        with open("data_collection/performance.txt", "a") as f:
            f.write(f"{processor()}: {sum(times)/len(times)}")
        times = []
    return times


if __name__ == "__main__":
    house_55 = StateOfObject(["bathroom", "largeroom", "smallroom"])
    os.makedirs("data_collection", exist_ok=True)
    performance_count_times = []
    # mqtt = MQTT_Publisher("username", "password", "broker_ip", "broker_port")
    mqtt = None
    if not os.path.exists("data_collection/iterations_logs.csv"):    
        file = open("data_collection/iterations_logs.csv", "a")
        file.write("time, energy_waste, temperature, shower_time, window_alert, is_shower, is_present\n")
        file.close()
    while True:
        begin = time()
        run(house_55, mqtt)
        time_of_loop = time()-begin
        print("time of loop: ", time_of_loop, "\n")
        performance_count_times = save_performance(time_of_loop, performance_count_times, 10)
        sleep(60)