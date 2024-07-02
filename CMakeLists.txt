cmake_minimum_required(VERSION 3.28)
project(CUDA_Raytracing LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 23)

add_executable(CUDA_Raytracing main.cu)

set_target_properties(CUDA_Raytracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
