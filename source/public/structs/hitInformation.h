#pragma once

#ifndef HITINFORMATION_H
#define HITINFORMATION_H

#include "color4.h"
#include "../materials/material.h"

/*
 *	HitInformation contains all information needed for a view ray: the closest hitDistance
 *	and the according hitPosition and the color of the object at this hitPosition.
 */

struct HitInformation
{
	float distance;
	vec3 normal;
	vec3 position;
	material* mat;

	__host__ __device__ HitInformation() { distance = INFINITY; position = vec3::zero(); normal = vec3::zero(); }
};

#endif
