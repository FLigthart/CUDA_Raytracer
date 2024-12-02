#pragma once

#ifndef QUAD_H
#define QUAD_H
#include "shape.h"

class quad : public Shape
{
public:
	__device__ quad(const vec3& position, const vec3& u, const vec3& v, material* mat)
		: u(u), v(v), mat(mat)
	{
		transform.position = Ray(position, vec3::zero());

		normal = cross(u, v);
		normal.normalize();

		setBoundingBox();
	}

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const override;
	__device__ aabb boundingBox() const override { return bbox; }

private:
	// The position is the middle of the quad.
	vec3 u;
	vec3 v;

	material* mat;
	aabb bbox;

	vec3 normal;

	__host__ __device__ vec3 GetQ() const; // Q is the origin of u and v

	__device__ void setBoundingBox();

	__device__ bool isInterior(float a, float b, HitInformation& hitInformation) const;
};

#endif
