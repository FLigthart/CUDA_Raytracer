#pragma once

#include "color4.h"

/*
 *	HitInformation contains all information needed for a view ray: the closest hitDistance
 *	and the according hitPosition and the color of the object at this hitPosition.
 */

struct HitInformation
{
	float hitDistance;
	vec3 hitPosition;
	color4 hitColor;

	__host__ __device__ HitInformation() { hitDistance = 0.0f; hitPosition = vec3::zero(); hitColor = color4::black(); }
};
