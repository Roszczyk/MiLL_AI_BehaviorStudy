#include <stdio.h>
#include "pico/stdlib.h"

int main() {
    stdio_init_all();
    printf("Hello, Raspberry Pi Pico!\n");

    while (true) {
        sleep_ms(1000);
        printf("Still running...\n");
    }
    return 0;
}