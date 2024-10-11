#pragma once

#ifndef INTErVAL_H
#define INTERVAL_H

#include <cmath>

struct interval
{
public:
	float min, max;

	__host__ __device__ interval()
	{
		min = -INFINITY;
		max = INFINITY;
	}

	__host__ __device__ interval(float _min, float _max)
	{
		min = _min;
		max = _max;
	}

	// Is between values (but not on values)
	__device__ bool surrounds(float x) const
	{
		return min < x && x < max;
	}
};

#endif
