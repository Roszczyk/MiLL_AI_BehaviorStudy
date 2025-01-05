#include "func.h"
#include "func.c"
#include "weights.h"
#include "data.h"

#include "esp_sleep.h"

float outputs[4];
int result;
RTC_DATA_ATTR int current = 0;

void forward_network(const float* input, float* output) {
    int hidden1_number = 120;
    int hidden2_number = 40;
    float hidden1[hidden1_number];
    float hidden2[hidden2_number];
    dense_layer(input, hidden1, INPUT_SIZE, hidden1_number);
    dense_1_layer(hidden1, hidden2, hidden1_number, hidden2_number);
    dense_2_layer(hidden2, output, hidden2_number, 4);
}


void setup() {
    Serial.begin(115200);

}

void loop() {
    forward_network(data_array[current], outputs);
    result = get_result_from_softmax(outputs, 4);
    Serial.print("Result: ");
    Serial.println(result);
    if (current >= DATA_ROWS) current = 0;
    else current++;

    esp_sleep_enable_timer_wakeup(5 * 1000000);
    esp_deep_sleep_start();
}
