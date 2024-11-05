import paho.mqtt.client as mqtt

class MQTT_Publisher():
    def __init__(self, username, password, broker_ip, broker_port):
        self.client = mqtt.Client()
        
        self.client.username_pw_set(username, password)
        self.client.connect(broker_ip, broker_port)

    def publish(self, topic, message, retain=True, qos=1):
        self.client.publish(topic, message, qos=qos, retain=retain)

    def publish_for_interface_joint(self, data):
        self.publish("shower_time", data["shower_time_score"])
        self.publish("energy_waste", data["energy_waste_score"])
        # self.publish("energy_day", data["daily_energy_sum"])
        self.publish("open_window_alarm", data["window_alert"])
        self.publish("temperature_comfort", data["temperature_score"])