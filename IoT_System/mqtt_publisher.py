import paho.mqtt.client as mqtt
from passwords_gitignore import get_mqtt_password

class MQTT_Publisher():
    def __init__(self, username, password, broker_ip, broker_port):
        self.client = mqtt.Client()
        
        self.client.username_pw_set(username, password)
        self.client.connect(broker_ip, broker_port)

    def publish(self, topic, message, retain=True, qos=1):
        self.client.publish(topic, message, qos=qos, retain=retain)

    def publish_for_interface_joint(self, data, house_nr):
        base_topic = f"{house_nr}/interface_score"
        self.publish(f"{base_topic}/shower_time", data["shower_time_score"])
        self.publish(f"{base_topic}/energy_waste", data["energy_waste_score"])
        self.publish(f"{base_topic}/energy_day", data["daily_energy_score"])
        self.publish(f"{base_topic}/open_window_alarm", data["window_alert"])
        self.publish(f"{base_topic}/temperature_comfort", data["temperature_score"])


def init_mqtt_publisher_for_wilga():
    return MQTT_Publisher("iot01", get_mqtt_password(), "10.89.10.1", 1883)