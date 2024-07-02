cmake_minimum_required(VERSION 3.28)
project(CUDA_Raytracing LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)

set(SOURCES source/private/main.cu)
set(HEADERS source/public/vec3.h
        source/public/ray.h
        source/public/camera.h)

add_executable(CUDA_Raytracing ${SOURCES})

set_target_properties(CUDA_Raytracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
