#include "data.h"
#include "func.h"

#include <stdio.h>
#include <stdlib.h>


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
    float outputs[4];
    int result = 0;
    for (int i=0; i<DATA_ROWS; i++){
        forward_network(data_array[i], outputs);
        result = get_result_from_softmax(outputs, 4);
        printf("NN Inference result: %d\n", result);
        // printf("%d ", result);
    }
}