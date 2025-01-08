cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(RPI_DT)

pico_sdk_init()

add_executable(RPI_DT main.c)

target_link_libraries(RPI_DT pico_stdlib)
target_link_libraries(RPI_DT pico_stdlib hardware_rtc)

pico_enable_stdio_usb(RPI_DT 1)
pico_enable_stdio_uart(RPI_DT 0)

pico_add_extra_outputs(RPI_DT)
