#pragma once

#include "math_constants.h"
#include "cuda_runtime_api.h"


class mathOperations
{
public:

	__host__ __device__ static float degreesToRadians(float degrees)
	{
		return degrees * CUDART_PI_F / 180.0f;
	}

	__host__ __device__ static float radiansToDegrees(float radians)
	{
		return radians * 180.0f / CUDART_PI_F;
	}
};