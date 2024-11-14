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

	__device__ interval(float _min, float _max)
	{
		min = _min;
		max = _max;
	}

	// Create an interval enclosing the two parameter intervals
	__device__ interval(const interval& a, const interval& b)
	{
		min = a.min <= b.min ? a.min : b.min;
		max = a.max >= b.max ? a.max : b.max;
	}

	__device__ float size() const
	{
		return max - min;
	}

	// Is between values (but not on values)
	__device__ bool surrounds(float x) const
	{
		return min < x && x < max;
	}

	__device__ float clamps(float x) const
	{
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}

	// "widen" interval by delta
	__device__ interval expand(float delta) const
	{
		float padding = delta / 2.0f;
		return interval(min - padding, max + padding);
	}

	// Returns interval from smallest to biggest value parameter
	__device__ static interval minToMax(float a, float b)
	{
		return (a <= b) ? interval(a, b) : interval(b, a);
	}

	__device__ static interval empty() { return interval(INFINITY, -INFINITY); }
	__device__ static interval universe() { return interval(-INFINITY, INFINITY); }
};

#endif
