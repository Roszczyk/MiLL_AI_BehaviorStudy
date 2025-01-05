#include <stdio.h>
#include <float.h>
#include <math.h>

#include "data.h"
#include "config.h"


double gaussian_likelihood(double x, double mean, double var) {
    if (var < 1e-6) {
        var = 1e-6;
    }
    return (1.0 / sqrt(2.0 * 3.14 * var)) * exp(-((x - mean) * (x - mean)) / (2.0 * var));
}


int predict(float input[INPUT_SIZE]) {
    double posteriors[CLASSES];
    double prior, likelihood;
    for (int i = 0; i < CLASSES; i++) {
        prior = log(priors[i]);
        likelihood = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            double likelihood_component = gaussian_likelihood(input[j], means[i][j], variances[i][j]);
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


int main(void){
    int result;
    for (int i=0; i<DATA_ROWS; i++){
        result = predict(data_array[i]);
        printf("%d ", result);
    }
}