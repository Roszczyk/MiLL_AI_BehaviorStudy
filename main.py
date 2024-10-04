from shower_time import is_shower_now, calculate_hour_for_shower
from data_acquisition import acquire_data_from_wilga
from interface_joint import do_calculating

from time import sleep, time
from datetime import datetime


class StateOfObject:
    def __init__(self):
        self.current_energy_sum = 0
        self.energy_before_today = None
        self.current_date = None

    def put_current_date(self):
        self.current_date = datetime.today().date()

    def reset_daily_energy_sum(self):
        self.energy_before_today = self.current_date
        self.current_energy_sum = 0


def run(state):
    if state.current_date != datetime.today().date():
        state.put_current_date()
        state.reset_daily_energy_sum()
    data = acquire_data_from_wilga(900)
    detect_shower = is_shower_now(data)
    best_shower_time = calculate_hour_for_shower(datetime.today().replace(hour=17, minute=0, second=0))
    score = do_calculating(data, best_shower_time)
    energy_waste_score = score["energy_waste_score"]
    temperature_score = score["temperature_score"]
    shower_time_score = score["shower_time_score"]
    window_alert = score["window_alert"]

    print(energy_waste_score, temperature_score, shower_time_score, window_alert, detect_shower)

if __name__ == "__main__":
    state = StateOfObject()
    while True:
        begin = time()
        run(state)
        print("time of loop: ", time()-begin)
        sleep(2)