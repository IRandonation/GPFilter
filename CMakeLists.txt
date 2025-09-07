# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(GPMPFilter)

set(CMAKE_CXX_STANDARD 17)
find_package(GTSAM REQUIRED)
message("-- GTSAM include directory: ${GTSAM_INCLUDE_DIR}")
message("-- GTSAM libraries: ${GTSAM_LIBRARIES}")

include_directories(
    include
    /usr/include/eigen3  # Eigen头文件路径
    ${GTSAM_INCLUDE_DIR}
)

add_executable(GPMPFilter
    src/main.cpp
    include/GPFactor.h
    include/MeasurementFactor.h
    include/VelocityConstraint.h
    include/AccelerationConstraint.h
)

# 手动指定 GTSAM 库路径
set(GTSAM_LIBS
    /usr/local/lib/libgtsam.so
    /usr/local/lib/libgtsam_unstable.so
)

target_link_libraries(GPMPFilter ${GTSAM_LIBS})