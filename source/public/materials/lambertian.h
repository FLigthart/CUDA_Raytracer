#pragma once

#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.h"
#include "../structs/hitInformation.h"
#include "../structs/vec3.h"
#include "../ray.h"
#include "../textures/texture.h"
#include "../textures/solidColor.h"

class lambertian : public material
{
public:
	__host__ __device__ explicit lambertian(const color4& alb) : texture(new solidColor(alb)) {}
	__host__ __device__ lambertian(texture* tex) : texture(tex) {}

	__device__ bool inline scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override;

	__host__ __device__ static lambertian* white() { return new lambertian(color4::white()); }

private:
	texture* texture;
};

__device__ bool lambertian::scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const
{
	vec3 target = hitInformation.position + hitInformation.normal + randomInUnitSphere(randomState);

	attenuation = texture->value(hitInformation.u, hitInformation.v, hitInformation.position);

	scattered = Ray(hitInformation.position, target - hitInformation.position, rayIn.time());

	return true;
}

#endif
