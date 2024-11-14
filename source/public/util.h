#pragma once

#include <cuda_runtime_api.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ );

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