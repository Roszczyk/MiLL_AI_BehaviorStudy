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


int main(void){
    int result;
    for (int i=0; i<DATA_ROWS; i++){
        // result = predict(args);
    }
}