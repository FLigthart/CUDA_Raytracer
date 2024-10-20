#pragma once

#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"
#include "../structs/hitInformation.h"
#include "../structs/vec3.h"
#include "../ray.h"

class lambertian : public material
{
public:
	__host__ __device__ explicit lambertian(const color4& alb) : albedo(alb) {}
	__host__ __device__ lambertian() : albedo(color4::white()) {}

	__device__ bool inline scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override;

	color4 albedo;

	__host__ __device__ static lambertian* white() { return new lambertian(color4::white()); }
};

__device__ bool lambertian::scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const
{
	vec3 target = hitInformation.position + hitInformation.normal + randomInUnitSphere(randomState);

	attenuation = albedo;

	scattered = Ray(hitInformation.position, target - hitInformation.position, rayIn.time());

	return true;
}

#endif
