#pragma once

#ifndef METAL_H
#define METAL_H

#include "material.h"
#include "../ray.h"
#include "../structs/color4.h"
#include "../structs/hitInformation.h"

class metal : public material
{
public:
	__host__ __device__ explicit metal(const color4& alb) : albedo(alb) {}

	__device__ bool scatter(const Ray& r_in, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override
	{
		vec3 reflected = reflect(r_in.direction(), hitInformation.normal);
		scattered = Ray(hitInformation.position, reflected);
		attenuation = albedo;
		return true;
	}

	color4 albedo;
};

#endif
