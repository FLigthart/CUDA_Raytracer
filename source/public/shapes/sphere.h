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
	aabb bbox;

	// Stationary Sphere
	__device__ Sphere(vec3 position, float _radius, material* _mat)
	{
		radius = _radius;
		transform = ShapeTransform(Ray(position, vec3::zero()));
		mat = _mat;

		vec3 sphereVector = vec3(radius, radius, radius);
		bbox = aabb(position - sphereVector, position + sphereVector);
	}

	// Moving Sphere
	__device__ Sphere(vec3 positionAtZero, vec3 positionAtOne, float _radius, material* _mat)
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

	__device__ bool checkIntersection(Ray& ray, interval hitInterval, HitInformation& hitInformation) const override;

	__device__ static void getSphereUv(const vec3& point, float& u, float& v);

	__device__ aabb boundingBox() const override { return bbox; }

	__device__ ~Sphere()
	{
		 delete mat;
	}
};

#endif
