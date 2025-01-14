#include "func.h"
#include "weights.h"
#include "data.h"
#include <math.h>

float relu(float x) {
    return x > 0 ? x : 0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void softmax(float* inputs, int length) {
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += exp(inputs[i]);
    }
    for (int i = 0; i < length; i++) {
        inputs[i] = exp(inputs[i]) / sum;
    }
}

void dense_layer(const float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_weights[j][i];
        }
        outputs[i] = relu(outputs[i]);
    }
}

void dense_1_layer(float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_1_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_1_weights[j][i];
        }
        outputs[i] = sigmoid(outputs[i]);
    }
}

void dense_2_layer(float * inputs, float * outputs, int number_inputs, int number_outputs){
    for (int i = 0; i<number_outputs; i++){
        outputs[i] = dense_2_bias[i];
        for (int j=0; j<number_inputs; j++){
            outputs[i] = outputs[i] + inputs[j]*dense_2_weights[j][i];
        }
    }
    softmax(outputs, 4);
}

void forward_network(const float* input, float* output) {
    int hidden1_number = 120;
    int hidden2_number = 40;
    float hidden1[hidden1_number];
    float hidden2[hidden2_number];
    dense_layer(input, hidden1, INPUT_SIZE, hidden1_number);
    dense_1_layer(hidden1, hidden2, hidden1_number, hidden2_number);
    dense_2_layer(hidden2, output, hidden2_number, 4);
}


int get_result_from_softmax(float * result_softmax, int lenght){
    int max = 0;
    for (int i=1; i<lenght; i++){
        if (result_softmax[max] < result_softmax[i]) max = i;
    }
    return max;
}
