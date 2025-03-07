cmake_minimum_required(VERSION 3.10)
project(rav)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# OpenMP handling - make it optional
option(USE_OPENMP "Use OpenMP for parallelization" ON)
set(HAVE_OPENMP 0)

if(USE_OPENMP)
    # Check if we're on macOS and using Apple Clang
    if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
        # Try to find OpenMP with Homebrew-installed libomp
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 3.9)
            set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
            set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
            
            if(EXISTS "${OPENMP_LIBRARIES}/libomp.dylib")
                message(STATUS "Found OpenMP via Homebrew libomp")
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OPENMP_INCLUDES}")
                set(OpenMP_C_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${OPENMP_LIBRARIES}/libomp.dylib")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OPENMP_INCLUDES}")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(HAVE_OPENMP 1)
            endif()
        endif()
    endif()
    
    # Standard OpenMP check
    if(NOT HAVE_OPENMP)
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(HAVE_OPENMP 1)
        else()
            message(STATUS "OpenMP not found or not supported by your compiler. Parallel processing will be disabled.")
        endif()
    endif()
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
if(HAVE_OPENMP AND EXISTS "${OPENMP_INCLUDES}")
    include_directories(${OPENMP_INCLUDES})
endif()

# Add executable
add_executable(rav main_real_time.cpp filters.cpp)

# Link libraries
target_link_libraries(rav ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

# Add OpenMP if available
if(HAVE_OPENMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rav OpenMP::OpenMP_CXX)
        add_definitions(-DUSE_OPENMP)
    elseif(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang" AND EXISTS "${OPENMP_LIBRARIES}/libomp.dylib")
        target_link_libraries(rav "${OPENMP_LIBRARIES}/libomp.dylib")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OPENMP)
    endif()
endif()

# Enable fast math optimizations
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(rav PRIVATE -ffast-math -ftree-vectorize)
endif()

# Display some information
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
if(HAVE_OPENMP)
    message(STATUS "OpenMP enabled")
    message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP disabled")
endif()
