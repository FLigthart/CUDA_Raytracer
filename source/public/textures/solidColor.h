#pragma once

#ifndef SOLID_COLOR_H
#define SOLID_COLOR_H

#include "texture.h"

class lambertian;

class solidColor : public texture
{
public:
	__host__ __device__ solidColor(const color4& albedo) : albedo(albedo) {}

	__host__ __device__ solidColor(float red, float green, float blue) : solidColor(color4(red, green, blue, 1.0f)) {}

	__device__ color4 value(float u, float v, const vec3& point) const override
	{
		return albedo;
	}

private:
	color4 albedo;
};


#endif