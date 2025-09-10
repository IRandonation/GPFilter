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
    src/GPInterpolator.cpp
    src/VelocityEstimator.cpp
    include/GPFactor.h
    include/MeasurementFactor.h
    include/VelocityConstraint.h
    include/AccelerationConstraint.h
    include/GPInterpolator.h
    include/VelocityEstimator.h
)

# 手动指定 GTSAM 库路径
set(GTSAM_LIBS
    /usr/local/lib/libgtsam.so
    /usr/local/lib/libgtsam_unstable.so
    /usr/local/lib/libmetis-gtsam.so
)

target_link_libraries(GPMPFilter ${GTSAM_LIBS})

# 设置 RPATH 以便运行时能找到 GTSAM 库
set_target_properties(GPMPFilter PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH /usr/local/lib
    BUILD_RPATH /usr/local/lib
)