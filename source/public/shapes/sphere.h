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

	// Stationary Sphere
	__host__ __device__ Sphere(vec3 position, float _radius, material* _mat)
	{
		radius = _radius;
		transform = ShapeTransform(Ray(position, vec3::zero()));
		mat = _mat;

		vec3 sphereVector = vec3(radius, radius, radius);
		bbox = aabb(position - sphereVector, position + sphereVector);
	}

	// Moving Sphere
	__host__ __device__ Sphere(vec3 positionAtZero, vec3 positionAtOne, float _radius, material* _mat)
	{
		radius = _radius;
		transform = ShapeTransform( Ray(positionAtZero, positionAtOne - positionAtZero));
		mat = _mat;

		// We want a bounding box for entire range of motion. So we take a box at time = 0 and time = 1, and compute a box around these boxes.
		vec3 sphereVector = vec3(radius, radius, radius);
		aabb box0(transform.position.at(0.0f) - sphereVector, transform.position.at(0.0f) + sphereVector);
		aabb box1(transform.position.at(1.0f) - sphereVector, transform.position.at(1.0f) + sphereVector);
		bbox = aabb(box0, box1);
	}

	__device__ bool inline checkIntersection(Ray& ray, interval hitInterval, HitInformation& hitInformation) const override;

	__device__ virtual aabb boundingBox() const override { return bbox; }

	__host__ __device__ ~Sphere()
	{
		 delete mat;
	}

private:
	aabb bbox;
};

__device__ bool Sphere::checkIntersection(Ray& ray, interval hitInterval, HitInformation& hitInformation) const
{
	// Calculate A, B and C for quadratic equation.
	vec3 currentCenter = transform.position.at(ray.time());
	vec3 centerToOrigin = currentCenter - ray.origin();

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

	hitInformation.normal = (hitInformation.position - currentCenter) / radius;

	hitInformation.mat = mat;

	return true;
}

#endif
