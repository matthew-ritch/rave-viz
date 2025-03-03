cmake_minimum_required(VERSION 3.10)
project(rav)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED )

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(rav main.cpp filters.cpp)

# Link libraries
target_link_libraries(rav ${OpenCV_LIBS})

# Enable optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
