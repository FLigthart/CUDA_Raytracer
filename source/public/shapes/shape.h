#pragma once

#include <cmath>

#include "../materials/lambertian.h"
#include "../structs/ShapeTransform.h"
#include "../structs/hitInformation.h"
#include "../structs/interval.h"
#include "../bvh/aabb.h"

class Ray;

class Shape
{

public:
	ShapeTransform transform;

	__device__ virtual bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const = 0;

	__device__ virtual aabb boundingBox() const = 0;

	__device__ static vec3 getFaceNormal(const Ray& r, const vec3& outwardNormal)
	{
		bool frontFace = dot(r.direction(), outwardNormal) < 0;
		return frontFace ? outwardNormal : -outwardNormal;
	}
};
