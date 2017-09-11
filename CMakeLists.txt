cmake_minimum_required(VERSION 3.1)

# Use C++11 standard
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

# Find OpenCV
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message("Found OpenCV version ${OpenCV_VERSION}")
endif()

# Main source and header files
file(GLOB SRC_SOURCES *.cpp )
file(GLOB SRC_HEADERS *.h *.hpp )

# Add source groups for clarity
source_group("Main Sources" FILES ${SRC_SOURCES})
source_group("Main Headers" FILES ${SRC_HEADERS})

# Build a list of all source files
list(APPEND ALL_SOURCES ${SRC_SOURCES})

# Build a list of all header files
list(APPEND ALL_HEADERS ${SRC_HEADERS})
        
include_directories(${OpenCV_INCLUDE_DIRS})
                
add_executable(flir-capture ${ALL_SOURCES} ${ALL_HEADERS})

target_link_libraries(flir-capture ${OpenCV_LIBS} flycapture)


