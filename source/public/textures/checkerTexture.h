#pragma once

#ifndef CHECKER_TEXTURE_H
#define CHECKER_TEXTURE_H

#include "solidColor.h"
#include "texture.h"

class solidColor;

class checkerTexture : public texture
{
public:
	__device__ checkerTexture(float scale, texture* even, texture* odd)
		: invScale(1.0f / scale), even(even), odd(odd) {}

	__device__ checkerTexture(float scale, const color4& c1, const color4& c2)
		: checkerTexture(scale, new solidColor(c1), new solidColor(c2)) { }

	__device__ color4 value(float u, float v, const vec3& point) const override
	{
		int xInt = static_cast<int>(std::floor(invScale * point.x()));
		int yInt = static_cast<int>(std::floor(invScale * point.y()));
		int zInt = static_cast<int>(std::floor(invScale * point.z()));

		bool isEven = (xInt + yInt + zInt) % 2 == 0;

		return isEven ? even->value(u, v, point) : odd->value(u, v, point);
	}

private:
	float invScale;
	texture* even;
	texture* odd;
};

#endif
