#pragma once

#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H
#include "../materials/material.h"
#include "../textures/solidColor.h"
#include "../textures/texture.h"

class diffuseLight : public material
{
public:
	__device__ diffuseLight(texture* texture) : tex(texture) {}
    __device__ diffuseLight(const color4& emit) : tex(new solidColor(emit)) {}

	__device__ color4 emitted(float u, float v, const vec3& point) const override
	{
		return tex->value(u, v, point);
	}

private:
	texture* tex;
};

#endif
