#pragma once

#ifndef SPHERE_H
#define SPHERE_H

#include "../../public/shapes/shape.h"
#include "../ray.h"
#include "../structs/hitInformation.h"

class Sphere : public Shape
{

public:

	float radius;
	material* mat;

	__host__ __device__ Sphere() { radius = 1.0f; transform = Transform(); mat = lambertian::white(); }
	__host__ __device__ Sphere(float _radius, material* _mat)
	{
		radius = _radius;
		transform = Transform();
		mat = _mat;
	}

	__device__ bool inline checkIntersection(Ray& ray, interval hitInterval, HitInformation& hitInformation) const override;

	 ~Sphere()
	 {
		 delete mat;
	 }
};

__device__ bool Sphere::checkIntersection(Ray& ray, interval hitInterval, HitInformation& hitInformation) const
{
	// Calculate A, B and C for quadratic equation.
	vec3 centerToOrigin = transform.position - ray.origin();

	// squaredLength is equivalent to the dot product of two equivalent vectors.
	float a = ray.direction().squaredLength();
	float b = dot(ray.direction(), centerToOrigin);
	float c = centerToOrigin.squaredLength() - radius * radius;

	float discriminant = b * b - a * c;

	// No intersection of ray with sphere
	if (discriminant < 0) return false;

	float dSquared = sqrt(discriminant);

	// Calculate the hit point of the ray on the sphere and the distance from the origin to the hit point.

	// One or Two hits. A different formula is used than the original quadratic formula, since if the squared discriminant and b are very close, catastrophic cancellation can happen.
	// The math from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html for the replacement formula is used here.

	float distance = (b - dSquared) / a;
	if (!hitInterval.surrounds(distance))
	{
		distance = (b + dSquared) / a;
		if (!hitInterval.surrounds(distance))
		{
			return false;
		}
	}

	hitInformation.distance = distance;

	hitInformation.position = ray.origin() + ray.direction() * hitInformation.distance;

	hitInformation.normal = (hitInformation.position - transform.position) / radius;

	hitInformation.mat = mat;

	return true;
}

#endif
