#pragma once

#include "../../public/shapes/shape.h"
#include "../ray.h"
#include "../structs/hitInformation.h"

class Sphere : public Shape
{

public:

	float radius;

	__host__ __device__ Sphere() { radius = 1.0f; transform = Transform(); color = color4::red(); }
	__host__ __device__ Sphere(float _radius)
	{
		radius = _radius;
		transform = Transform();
		color = color4::red();
	}

	__device__ bool inline checkIntersection(Ray& ray, HitInformation& hitInformation) const override;
};

__device__ bool Sphere::checkIntersection(Ray& ray, HitInformation& hitInformation) const
{
	// Calculate A, B and C for quadratic equation.
	vec3 centerToOrigin = ray.origin() - transform.position;

	// squaredLength is equivalent to the dot product of two equivalent vectors.
	float a = ray.direction().squaredLength();
	float b = dot(2 * ray.direction(), centerToOrigin);
	float c = centerToOrigin.squaredLength() - radius * radius;

	float discriminant = b * b - 4 * a * c;

	// No intersection of ray with sphere
	if (discriminant < 0) return false;

	// Calculate the hit point of the ray on the sphere and the distance from the origin to the hit point.

	// One or Two hits. A different formula is used than the original quadratic formula, since if the squared discriminant and b are very close, catastrophic cancellation can happen.
	// The math from https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection.html for the replacement formula is used here.

	float q = b < 0 ?
		-0.5f * (b + sqrt(discriminant)) :
		-0.5f * (b - sqrt(discriminant));

	float hitDistanceOne = q / a;
	float hitDistanceTwo = c / q;

	// If hitDistanceOne is smaller or equal to hitDistanceTwo, hitPointOne is the first hit point of the ray with the sphere. Otherwise, it is hitDistanceTwo.
	// Use the closest one to calculate hitPointOne, since the camera can see that one.
	hitInformation.distance = (hitDistanceOne <= hitDistanceTwo) ? hitDistanceTwo : hitDistanceOne;

	hitInformation.position = ray.origin() + ray.direction() * hitInformation.distance;

	hitInformation.color = color;

	hitInformation.normal = (hitInformation.position - transform.position) / radius;

	return true;
}

