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

	__device__ aabb(const interval& x, const interval& y, const interval& z)
		: x(x), y(y), z(z) {}

	// a and b are extrema for the bounding box.
	__device__ aabb(const vec3& a, const vec3& b)
	{
		x = interval::minToMax(a.x(), b.x());
		y = interval::minToMax(a.y(), b.y());
		z = interval::minToMax(a.z(), b.z());
	}

	__device__ aabb(const aabb& box0, const aabb& box1)
	{
		x = interval(box0.x, box1.x);
		y = interval(box0.y, box1.y);
		z = interval(box0.z, box1.z);
	}

	__device__ const interval& axisInterval(int n) const
	{
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__device__ bool checkIntersection (const Ray& ray, interval rayT) const
	{
		const vec3& rayOrigin = ray.origin();
		const vec3& rayDirection = ray.direction();

		for (int axis = 0; axis < 3; axis++)
		{
			const interval& ax = axisInterval(axis);
			const float adinv = 1.0f / rayDirection[axis];

			float t0 = (ax.min - rayOrigin[axis]) * adinv;
			float t1 = (ax.max - rayOrigin[axis]) * adinv;

			if (t0 < t1) {
				if (t0 > rayT.min) rayT.min = t0;
				if (t1 < rayT.max) rayT.max = t1;
			}
			else {
				if (t1 > rayT.min) rayT.min = t1;
				if (t0 < rayT.max) rayT.max = t0;
			}

			if (rayT.max <= rayT.min)
				return false;
		}

		return true;
	}

	__device__ int longestAxis() const
	{
		if (x.size() > z.size())
			return x.size() > z.size() ? 0 : 2;
		else
			return y.size() > z.size() ? 1 : 2;
	}

	static const aabb empty, universe;
};

const aabb aabb::empty = aabb(interval::empty, interval::empty, interval::empty);
const aabb aabb::universe = aabb(interval::universe, interval::universe, interval::universe);

#endif
