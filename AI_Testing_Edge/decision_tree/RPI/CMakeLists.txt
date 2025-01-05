cmake_minimum_required(VERSION 3.13)

# Nazwa projektu
project(RPI)
enable_language(ASM)

# Ustaw standardy C i C++
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Ustaw ścieżki do kompilatora
set(CMAKE_C_COMPILER "C:/Program Files (x86)/GNU Arm Embedded Toolchain/10 2021.10/bin/arm-none-eabi-gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/GNU Arm Embedded Toolchain/10 2021.10/bin/arm-none-eabi-g++.exe")
set(CMAKE_ASM_COMPILER "C:/Program Files (x86)/GNU Arm Embedded Toolchain/10 2021.10/bin/arm-none-eabi-gcc.exe")

# Ustaw ścieżkę do SDK
include("C:/Users/roszc/Desktop/desk/pico/pico-sdk/external/pico_sdk_import.cmake")

# Określ płytkę
set(PICO_BOARD pico_w)

# Inicjalizacja SDK
pico_sdk_init()

# Dodaj plik źródłowy
add_executable(RPI main.c)

# Połącz z bibliotekami SDK
target_link_libraries(RPI pico_stdlib)

# Dodaj dodatkowe wyjścia (np. UF2)
pico_add_extra_outputs(RPI)