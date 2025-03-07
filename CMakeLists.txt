cmake_minimum_required(VERSION 3.10)
project(rav)

# Enable C++11 standard for thread support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add optimization flags but without OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(rav main_real_time.cpp filters.cpp)

# Link libraries - use standard threading library instead of OpenMP
target_link_libraries(rav ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Enable fast math optimizations
if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(rav PRIVATE -ffast-math -ftree-vectorize)
endif()
