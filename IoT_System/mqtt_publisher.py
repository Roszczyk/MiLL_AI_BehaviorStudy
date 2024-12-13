import paho.mqtt.client as mqtt
from passwords_gitignore import get_mqtt_password
from time import sleep

def ping(host):
    import subprocess
    try:
        cmd = ["ping", "-n", "1", host] if subprocess.os.name == "nt" else ["ping", "-c", "1", host]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        output = result.stdout.lower()
        if result.returncode == 1 or "unreachable" in output:
            return False
        return True
        
    except Exception as e:
        print(f"Błąd: {e}")
        return False


class MQTT_Publisher():
    def __init__(self, username, password, broker_ip, broker_port):
        self.client = mqtt.Client()
        self.ip = broker_ip
        
        self.client.username_pw_set(username, password)
        self.client.connect(broker_ip, broker_port)

    def publish(self, topic, message, retain=True, qos=1):
        self.client.publish(topic, message, qos=qos, retain=retain)

    def publish_for_interface_joint(self, data, house_nr, iteration=10, pings=2):
        base_topic = f"{house_nr}/interface_score"
        try: 
            self.publish(f"{base_topic}/shower_time", data["shower_time_score"])
            self.publish(f"{base_topic}/energy_waste", data["energy_waste_score"])
            self.publish(f"{base_topic}/energy_day", data["daily_energy_score"])
            self.publish(f"{base_topic}/open_window_alarm", data["window_alert"])
            self.publish(f"{base_topic}/temperature_comfort", data["temperature_score"])
        except Exception as e:
            print(f"Exception: {e}\nTrying to publish to MQTT, remaining attempts: {iteration}")
            while True:
                sleep(60)
                print(f"trying to publish to MQTT, checking if host is reachable, pings: {pings}")
                is_reachable = ping(self.broker_ip)
                if not is_reachable and pings <= 0:
                    print(f"Host {self.broker_ip} is unreachable, unable to publish to MQTT broker")
                    return False
                if is_reachable:
                    print(f"Host {self.broker_ip} is reachable, trying to publish to MQTT broker")
                    break
            if iteration<=0:
                print("Connection to broker MQTT impossible, exception: ", e)
                return False
            self.publish_for_interface_joint(data, house_nr, iteration-1)


def init_mqtt_publisher_for_wilga():
    return MQTT_Publisher("iot01", get_mqtt_password(), "10.89.10.1", 1883)