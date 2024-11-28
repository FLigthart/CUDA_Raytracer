#pragma once

#ifndef IMAGETEXTURE_H
#define IMAGETEXTURE_H

#include "texture.h"

class imageTexture : public texture
{
public:

	__host__ imageTexture(const char* filePath);

	__device__ imageTexture(unsigned char* data, int w, int h, int c);

	__host__ __device__ ~imageTexture();

	__device__ color4 value(float u, float v, const vec3& point) const override;

	int width;
	int height;
	int channels = 3;
	unsigned char* pixelData;
	int pixelDataSize;
};

#endif