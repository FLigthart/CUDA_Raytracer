#pragma once

#include "../structs/interval.h"
#include "../structs/vec3.h"
#include "../ray.h"

#ifndef AABB_H
#define AABB_H

class aabb
{
public:
	interval x, y, z;

	aabb() = default; // The default AABB is empty

	__host__ __device__ aabb(const interval& x, const interval& y, const interval& z)
		: x(x), y(y), z(z)
	{
		padToMinimums(); //Ensures that volume is never 0 to prevent unexpected behaviour with intersections
	}

	// a and b are extrema for the bounding box.
	__host__ __device__ aabb(const vec3& a, const vec3& b)
	{
		x = interval::minToMax(a.x(), b.x());
		y = interval::minToMax(a.y(), b.y());
		z = interval::minToMax(a.z(), b.z());

		padToMinimums();
	}

	__host__ __device__ aabb(const aabb& box0, const aabb& box1)
	{
		x = interval(box0.x, box1.x);
		y = interval(box0.y, box1.y);
		z = interval(box0.z, box1.z);
	}

	__host__ __device__ const interval& axisInterval(int n) const
	{
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__host__ __device__ bool checkIntersection (const Ray& ray, interval rayT) const
	{
		const vec3& rayOrigin = ray.origin();
		const vec3& rayDirection = ray.direction();

		for (int axis = 0; axis < 3; axis++)
		{
			const interval& ax = axisInterval(axis);
			const float adinv = 1.0f / rayDirection[axis];

			float t0 = (ax.min - rayOrigin[axis]) * adinv;
			float t1 = (ax.max - rayOrigin[axis]) * adinv;

			if (t0 < t1)
			{
				if (t0 > rayT.min) rayT.min = t0;
				if (t1 < rayT.max) rayT.max = t1;
			}
			else 
			{
				if (t1 > rayT.min) rayT.min = t1;
				if (t0 < rayT.max) rayT.max = t0;
			}

			if (rayT.max <= rayT.min)
				return false;
		}

		return true;
	}

	__host__ __device__ int longestAxis() const
	{
		if (x.size() > z.size())
			return x.size() > z.size() ? 0 : 2;
		else
			return y.size() > z.size() ? 1 : 2;
	}

	// Adjust the AABB so that the volume of the AABB > 0
	__host__ __device__ void padToMinimums()
	{
		float delta = 0.0001f;
		if (x.size() < delta) x = x.expand(delta);
		if (y.size() < delta) y = y.expand(delta);
		if (z.size() < delta) z = z.expand(delta);
 	}

	__host__ __device__ static aabb empty() { return aabb(interval::empty(), interval::empty(), interval::empty()); }
	__host__ __device__ static aabb universe() { return aabb(interval::universe(), interval::universe(), interval::universe()); }
};

#endif
