#include <float.h>
#include <math.h>
#include "esp_sleep.h"

#include "data.h"
#include "config.h"

RTC_DATA_ATTR int current = 0;
int result;


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

void setup() {
  Serial.begin(115200);

}

void loop() {
    result = predict(data_array[current]);
    Serial.print("Result: ");
    Serial.println(result);
    if (current >= DATA_ROWS) current = 0;
    else current++;

    esp_sleep_enable_timer_wakeup(5 * 1000000);
    esp_deep_sleep_start();
}
