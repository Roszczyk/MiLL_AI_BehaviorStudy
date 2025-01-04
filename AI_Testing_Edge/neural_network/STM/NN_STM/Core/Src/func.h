
#ifndef FUNC_H
#define FUNC_H

float relu(float x);
float sigmoid(float x);
void softmax(float* inputs, int length);
void dense_layer(const float * inputs, float * outputs, int number_inputs, int number_outputs);
void dense_1_layer(float * inputs, float * outputs, int number_inputs, int number_outputs);
void dense_2_layer(float * inputs, float * outputs, int number_inputs, int number_outputs);
void forward_network(const float* input, float* output);
int get_result_from_softmax(float * result_softmax, int lenght);

#endif
