#pragma once

#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include <cub/util_macro.cuh>

#include "shape.h"
#include "../lights/isotropic.h"

class constantMedium : public Shape
{
public:
	__device__ constantMedium(Shape* boundary, float density, texture* tex)
		: boundary(boundary), negativeInvDensity(-1 / density),
		  phaseFunction(new isotropic(tex))
	{}

	__device__ constantMedium(Shape* boundary, float density, const color4& albedo)
		: boundary(boundary), negativeInvDensity(-1 / density),
		phaseFunction(new isotropic(albedo))
	{}

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const override
	{
		HitInformation recordOne, recordTwo;

		if (!boundary->checkIntersection(ray, interval::universe(), recordOne, localRandomState))
			return false;

		if (!boundary->checkIntersection(ray, interval(recordOne.distance + VERY_SMALL_NUMBER, INFINITY), recordTwo, localRandomState))
			return false;

		recordOne.distance = cub::max(recordOne.distance, hitRange.min);
		recordTwo.distance = cub::min(recordTwo.distance, hitRange.max);

		if (recordOne.distance >= recordTwo.distance)
			return false;

		recordOne.distance = cub::max(recordOne.distance, 0.0f);

		float rayLength = ray.direction().length();
		float distanceInsideBoundary = (recordTwo.distance - recordOne.distance) * rayLength;
		float hitDistance = negativeInvDensity * log(curand_uniform(localRandomState));

		if (hitDistance > distanceInsideBoundary)
			return false;

		hitInformation.distance = recordOne.distance + hitDistance / rayLength;
		hitInformation.position = ray.at(hitInformation.distance);

		hitInformation.normal = vec3(1.0f, 0.0f, 0.0f);
		hitInformation.mat = phaseFunction;

		return true;
	}

	__device__ aabb boundingBox() const override { return boundary->boundingBox(); }

private:
	Shape* boundary;
	float negativeInvDensity;
	material* phaseFunction;
};

#endif