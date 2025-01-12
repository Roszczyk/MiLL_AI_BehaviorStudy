cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(RPI_Bayes)

pico_sdk_init()

add_executable(RPI_Bayes main.c)

target_sources(RPI_Bayes PRIVATE data.c data.h config.c config.h)

target_link_libraries(RPI_Bayes pico_stdlib pico_time)

pico_enable_stdio_usb(RPI_Bayes 1)
pico_enable_stdio_uart(RPI_Bayes 0)

pico_add_extra_outputs(RPI_Bayes)
