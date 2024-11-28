#pragma once

#ifndef NOISE_TEXTURE_H
#define NOISE_TEXTURE_H

#include "perlin.h"
#include "texture.h"

class noiseTexture : public texture
{
public:
	__device__ noiseTexture(float sc, curandState* localRandomState) : noise(localRandomState), scale(sc) {}

	__device__ color4 value(float u, float v, const vec3& point) const override;

private:
	perlin noise;
	float scale;
};
#endif