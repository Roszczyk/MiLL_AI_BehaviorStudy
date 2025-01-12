cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(RPI_NN)

pico_sdk_init()

add_executable(RPI_NN main.c)

target_sources(RPI_NN PRIVATE data.c data.h func.c func.h weights.c weights.h)

target_link_libraries(RPI_NN pico_stdlib pico_time)

pico_enable_stdio_usb(RPI_NN 1)
pico_enable_stdio_uart(RPI_NN 0)

pico_add_extra_outputs(RPI_NN)
