#include "data.h"
#include "func.h"

#include <stdio.h>
#include <stdlib.h>


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