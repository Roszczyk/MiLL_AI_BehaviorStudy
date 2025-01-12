#include "data.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "pico/stdlib.h"
#include "pico/time.h"

#define INPUT_SIZE DATA_ITEMS

double gaussian_likelihood(double x, double mean, double var) {
    if (var < 1e-6) {
        var = 1e-6;
    }
    return (1.0 / sqrt(2.0 * 3.14 * var)) * exp(-((x - mean) * (x - mean)) / (2.0 * var));
}


int predict(float input[INPUT_SIZE]) {
    double posteriors[CLASSES];
    double prior, likelihood;
    double likelihood_component;
    for (int i = 0; i < CLASSES; i++) {
        prior = log(priors[i]);
        likelihood = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            likelihood_component = gaussian_likelihood(input[j], means[i][j], variances[i][j]);
            if (likelihood_component <= 0) {
                likelihood += -DBL_MAX; 
            } else {
                likelihood += log(likelihood_component);
            }
        }
        posteriors[i] = likelihood + prior;
        }
        int result = 0;
        for (int i = 1; i < CLASSES; i++) {
            if (posteriors[i] > posteriors[result]) {
                result = i;
            }
        }
        return result;
}


int main() {
    stdio_init_all();
    sleep_ms(2000);

    int result;
    int current = 0;
    while (true) {
        result = predict(data_array[current]);
        printf("Result: %d\n", result);
        if (current >= DATA_ROWS) current = 0;
        else current++;
        sleep_ms(5*1000);
    }
    return 0;
}