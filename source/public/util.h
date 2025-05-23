#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ );

#define VERY_SMALL_NUMBER 0.00000001f // Useful for comparisons due to floating number inaccuracy

#define INIT_LIST_AND_TREE(size)                                                   \
    listSize = (size);                                                             \
    checkCudaErrors(cudaMalloc((void **)&d_shapeList, listSize * sizeof(Shape*))); \
    treeSize = 2 * listSize;                                                       \
    h_bvhTree = new bvhNode[treeSize];                                             \
    checkCudaErrors(cudaMalloc((void **)&d_bvhTree, treeSize * sizeof(bvhNode)));

#define RND (curand_uniform(&localRandomState))

#define INIT_RAND_LOCAL()         \
    curandState localRandomState; \
    curand_init(randomSeed, 0, 0, &localRandomState);

inline void check_cuda(cudaError_t result, char const* const function, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << function << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

__host__ __device__ static float minimum(float one, float two)
{
    return one < two ? one : two;
}

__host__ __device__ static float maximum(float one, float two)
{
    return one > two ? one : two;
}