#pragma once

#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "../ray.h"
#include "../structs/color4.h"
#include "../structs/hitInformation.h"

class dielectric : public material
{
public:
	__host__ __device__ explicit dielectric(float _refractionIndex) : refractionIndex(_refractionIndex)
	{
		
	}

	__device__ bool scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override
	{
		vec3 outwardNormal;
		vec3 reflected = reflect(rayIn.direction(), hitInformation.normal);
		float etaiOverEtat;


		attenuation = color4(1.0f, 1.0f, 1.0f, 1.0f);
		vec3 refracted;
		float reflectProbability;
		float cosine;

		float dotRayInNormal = dot(rayIn.direction(), hitInformation.normal);
		if (dotRayInNormal > 0.0f)
		{
			outwardNormal = -hitInformation.normal;
			etaiOverEtat = refractionIndex;
			cosine = dotRayInNormal / rayIn.direction().length();
			cosine = sqrt(1.0f - refractionIndex * refractionIndex * (1 - cosine * cosine));
		}
		else
		{
			outwardNormal = hitInformation.normal;
			etaiOverEtat = 1.0f / refractionIndex;
			cosine = -dotRayInNormal / rayIn.direction().length();
		}

		if (refract(rayIn.direction(), outwardNormal, etaiOverEtat, refracted))
		{
			reflectProbability = schlick(cosine, refractionIndex);
		}
		else
		{
			reflectProbability = 1.0f;
		}

		// Reflected ray
		if (curand_uniform(randomState) < reflectProbability)
		{
			scattered = Ray(hitInformation.position, reflected, rayIn.time());
		}
		// Refracted ray
		else
		{
			scattered = Ray(hitInformation.position, refracted, rayIn.time());
		}

		return true;
	}

private:
	// Refractive index in vacuum or air, or the ratio of the material's refractive index over
	// the refractive index of the enclosing media. https://en.wikipedia.org/wiki/Refractive_index

	/*
	 * Given a sphere of material with an index of refraction greater than air,
	 * there's no incident angle that will yield total internal reflection neither at the ray-sphere entrance point nor at the ray exit.
	 * This is due to the geometry of spheres, as a grazing incoming ray will always be bent to a smaller angle,
	 * and then bent back to the original angle on exit.
	 */
	float refractionIndex; 
};

#endif
