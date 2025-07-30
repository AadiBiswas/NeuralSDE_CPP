cmake_minimum_required(VERSION 3.10)

project(NeuralSDE_CPP)

set(CMAKE_CXX_STANDARD 17)

# Existing header dirs
include_directories(include)

# Add tiny-dnn (header-only)
include_directories(external/tiny-dnn)

# Existing sim executable
add_executable(NeuralSDE_CPP src/main.cpp src/data_loader.cpp)

# New trainer executable
add_executable(trainer src/train.cpp src/data_loader.cpp)
