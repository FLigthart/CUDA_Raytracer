#pragma once

#include "vec3.h"

struct Transform
{

public:

	vec3 position;
	vec3 rotation;
	vec3 scale;

	// Constructor. Default values in case you want an "empty" Transform.
	__host__ __device__ explicit Transform(vec3 _position = vec3::zero(), vec3 _rotation = vec3::zero(), vec3 _scale = vec3::one()) { position = _position; rotation = _rotation; scale = _scale; }
};
