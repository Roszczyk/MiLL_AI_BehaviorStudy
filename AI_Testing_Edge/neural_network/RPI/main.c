#include "data.h"
#include "weights.h"
#include "func.h"

#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/time.h"


void forward_network(float* input, float* output) {
    int hidden1_number = 120;
    int hidden2_number = 40;
    float hidden1[hidden1_number];
    float hidden2[hidden2_number];
    dense_layer(input, hidden1, INPUT_SIZE, hidden1_number);
    dense_1_layer(hidden1, hidden2, hidden1_number, hidden2_number);
    dense_2_layer(hidden2, output, hidden2_number, 4);
}


int main() {
    stdio_init_all();
    sleep_ms(2000);
    
    float outputs[4];
    int result;
    int current = 0;

    while (true) {
        forward_network(data_array[current], outputs);
        result = get_result_from_softmax(outputs, 4);
        printf("Result: %d\n", result);
        if (current >= DATA_ROWS) current = 0;
        else current++;
        sleep_ms(5*1000);
    }
    return 0;
}