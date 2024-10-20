#pragma once

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "vec3.h"

struct ShapeTransform
{

public:

	// Center is a ray from position at time 0 to position at time 1. This is done for Motion Blur.
	Ray position;
	vec3 rotation;
	vec3 scale;

	// Constructor. Default values in case you want an "empty" ShapeTransform.
	__host__ __device__ explicit ShapeTransform(const Ray& _position = Ray::zero(), vec3 _rotation = vec3::zero(), vec3 _scale = vec3::one()) { position = _position; rotation = _rotation; scale = _scale; }
};

#endif
