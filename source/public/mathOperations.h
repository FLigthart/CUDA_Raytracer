#pragma once

#ifndef MATHOPERATIONS_H
#define MATHOPERATIONS_H

#include "math_constants.h"


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

	__host__ __device__ static float setFloatWithinBounds(float value, float min, float max)
	{
		if (value < min)
		{
			return min;
		}

		if (value > max)
		{
			return max;
		}

		return value;
	}
};

#endif
