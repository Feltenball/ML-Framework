cmake_minimum_required(VERSION 3.29)
project(Tensors)

set(CMAKE_CXX_STANDARD 20)

add_executable(Tensors main.cpp
        Tensor.h
        Tensor.cpp
        Graph.h
        Operation.h
        Operation.cpp
        Graph.cpp)
