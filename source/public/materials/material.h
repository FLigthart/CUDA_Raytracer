#pragma once

#include <curand_kernel.h>

#ifndef MATERIAL_H
#define MATERIAL_H

struct color4;
struct HitInformation;
class Ray;

class material
{
public:
	virtual ~material() = default;

	__device__ virtual bool scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const = 0;

protected:
	__device__ static float schlick(float cosine, float refractionIndex)
	{
		float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
	}
};

#endif