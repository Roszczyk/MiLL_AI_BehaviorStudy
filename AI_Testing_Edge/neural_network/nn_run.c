#include "data.h"
#include "model.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float relu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float* inputs, int length) {
    float sum = 0;
    printf("sum: %f\n", sum);
    for (int i = 0; i < length; i++) {
        sum += exp(inputs[i]);
    }
    for (int i = 0; i < length; i++) {
        inputs[i] = exp(inputs[i]) / sum;
    }
}

void dense_layer(float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_weights[j][i];
        }
        outputs[i] = relu(outputs[i]);
        printf("1 Output %d: %f\n", i, outputs[i]);
    }
}

void dense_1_layer(float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_1_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_1_weights[j][i];
        }
        outputs[i] = sigmoid(outputs[i]);
        printf("1 Output %d: %f\n", i, outputs[i]);
    }
}

void dense_2_layer(float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_2_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_2_weights[j][i];
        }
        outputs[i] = sigmoid(outputs[i]);
        printf("2 Output %d: %f\n", i, outputs[i]);
    }
    softmax(outputs, 4);
    printf("softmax: %f %f %f %f\n", outputs[0], outputs[1], outputs[2], outputs[3]);
}

void forward_network(float* input, float* output) {
    int hidden1_number = 120;
    int hidden2_number = 40;
    float hidden1[hidden1_number];
    float hidden2[hidden2_number];
    dense_layer(input, hidden1, INPUT_SIZE, hidden1_number);
    dense_1_layer(hidden1, hidden2, hidden1_number, hidden2_number);
    dense_2_layer(hidden2, output, hidden1_number, 4);
}

int get_result_from_softmax(float * result_softmax, int lenght){
    int max = 0;
    for (int i=1; i<lenght; i++){
        if (result_softmax[max] < result_softmax[i]) max = i;
    }
    return max;
}

int main() {
    float outputs[4];
    int result = 0;
    for (int i=0; i<1; i++){
        forward_network(data_array[i], outputs);
        result = get_result_from_softmax(outputs, 4);
        printf("Inference result: %d\n", result);
    }
}