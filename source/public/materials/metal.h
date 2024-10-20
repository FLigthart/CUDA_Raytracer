#pragma once

#ifndef METAL_H
#define METAL_H

#include "material.h"
#include "../ray.h"
#include "../structs/color4.h"
#include "../structs/hitInformation.h"
#include "../mathOperations.h"

class metal : public material
{
public:
	__host__ __device__ explicit metal(const color4& alb, float _fuzz) : albedo(alb)
	{
		fuzz = mathOperations::setFloatWithinBounds(_fuzz, 0.0f, 1.0f);
	}

	__device__ bool scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override
	{
		vec3 reflected = reflect(rayIn.direction(), hitInformation.normal);
		reflected = reflected.normalized() + (fuzz * randomInUnitSphere(randomState));	// Add fuzz by giving the ray a random offset at its endpoint.
		scattered = Ray(hitInformation.position, reflected, rayIn.time());
		attenuation = albedo;
		return (dot(scattered.direction(), hitInformation.normal) > 0);
	}

	color4 albedo;
	float fuzz;	// Blurriness of sphere in range 0 - 1
};

#endif
