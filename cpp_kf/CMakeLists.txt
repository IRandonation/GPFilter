cmake_minimum_required(VERSION 3.10)
project(GPFilterKF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(gpfilter_kf
    src/main.cpp
)

target_include_directories(gpfilter_kf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Enable more warnings for safer code
if (MSVC)
    target_compile_options(gpfilter_kf PRIVATE /W4)
else()
    target_compile_options(gpfilter_kf PRIVATE -Wall -Wextra -Wpedantic)
endif()