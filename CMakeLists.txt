# Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
# This file is part of libdeepvac, licensed under the GPLv3 (the "License")
# You may not use this file except in compliance with the License.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pydeepvac
    VERSION 0.1.0
    DESCRIPTION "A library for using PyTorch model in C++."
    HOMEPAGE_URL "https://github.com/deepvac/libdeepvac"
    LANGUAGES CXX)

option(GARRULOUS_GEMFIELD "Build deepvac libraries garrulously" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

#setup library output dir
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#setup binary output dir
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

file(GLOB CPP4PYTHON_LIST src/*.cpp)
message(STATUS "found CPP4PYTHON_LIST: " ${CPP4PYTHON_LIST})
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Deepvac REQUIRED)

if(GARRULOUS_GEMFIELD)
    message(STATUS "will build pydeepvac library garrulously...")
    add_definitions(-DGARRULOUS_GEMFIELD)
endif()

message(STATUS "TORCH_LIBRARIES " ${TORCH_LIBRARIES})
message(STATUS "OpenCV_LIBS " ${OpenCV_LIBS})

add_executable( pydeepvac ${CPP4PYTHON_LIST} )
target_link_libraries( pydeepvac deepvac ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(pydeepvac PROPERTIES LINK_FLAGS_RELEASE -s)
target_include_directories(pydeepvac PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

