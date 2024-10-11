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

	__device__ virtual bool scatter(const Ray& r_in, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const = 0;
};

#endif