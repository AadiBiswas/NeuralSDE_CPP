cmake_minimum_required(VERSION 3.10)

project(NeuralSDE_CPP)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(NeuralSDE_CPP src/main.cpp src/data_loader.cpp)

