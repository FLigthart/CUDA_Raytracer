#pragma once

#include "color4.h"

/*
 *	HitInformation contains all information needed for a view ray: the closest hitDistance
 *	and the according hitPosition and the color of the object at this hitPosition.
 */

struct HitInformation
{
	float distance;
	vec3 normal;
	vec3 position;
	color4 color;

	__host__ __device__ HitInformation() { distance = INFINITY; position = vec3::zero(); color = color4::black(); normal = vec3::zero(); }
};
