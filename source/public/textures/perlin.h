#pragma once

#ifndef PERLIN_TEXTURE_H
#define PERLIN_TEXTURE_H

#include <curand_kernel.h>

struct vec3;

class perlin
{
public:
	__device__ perlin(curandState* localRandState);

	__device__ ~perlin();

	__device__ float noise(const vec3& p) const;

	__device__ float turb(const vec3& p, int depth = 7) const;

private:
	static const int pointCount = 256;

	vec3* randomVec;

	int* permX;
	int* permY;
	int* permZ;

	__device__ static int* perlinGeneratePerm(curandState* localRandomState);

	__device__ static void permute(int* p, int n, curandState * localRandomState);

	__device__ static float perlinInterpolate(vec3 c[2][2][2], float u, float v, float w);
};

#endif
