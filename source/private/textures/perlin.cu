#include "../../public/textures/perlin.h"
#include "../../public/structs/vec3.h"

__device__ perlin::perlin(curandState* localRandState)
{
	randomVec = new vec3[perlin::pointCount];
	for (int i = 0; i < pointCount; i++)
	{
		vec3 randomVector = random(localRandState);
		randomVec[i] = vec3(-0.5f + randomVector.x(), -0.5f + randomVector.y(), -0.5f + randomVector.z()) * 2;
	}

	permX = perlinGeneratePerm(localRandState);
	permY = perlinGeneratePerm(localRandState);
	permZ = perlinGeneratePerm(localRandState);
}

__device__ perlin::~perlin()
{
	delete[] randomVec;
	delete[] permX;
	delete[] permY;
	delete[] permZ;
}

__device__ float perlin::noise(const vec3& p) const
{
	float u = p.x() - floor(p.x());
	float v = p.y() - floor(p.y());
	float w = p.z() - floor(p.z());

	int i = static_cast<int>(floor(p.x()));
	int j = static_cast<int>(floor(p.y()));
	int k = static_cast<int>(floor(p.z()));
	vec3 c[2][2][2];

	for (int di = 0; di < 2; di++)
	{
		for (int dj = 0; dj < 2; dj++)
		{
			for (int dk = 0; dk < 2; dk++)
			{
				c[di][dj][dk] = 
					randomVec[permX[(i + di) & 255] ^ 
					permY[(j + dj) & 255] ^ 
					permZ[(k + dk) & 255]];
			}
		}
	}

	return perlinInterpolate(c, u, v, w);
}

__device__ float perlin::turb(const vec3& p, int depth) const
{
	float acc = 0.0f;
	vec3 tempP = p;
	float weight = 1.0f;

	for (int i = 0; i < depth; i++)
	{
		acc += weight * noise(tempP);
		weight *= 0.5f;
		tempP *= 2;
	}

	return fabs(acc);
}

__device__ int* perlin::perlinGeneratePerm(curandState* localRandomState)
{
	int* p = new int[pointCount];

	for (int i = 0; i < pointCount; i++)
	{
		p[i] = i;
	}

	permute(p, pointCount, localRandomState);

	return p;
}

__device__ void perlin::permute(int* p, int n, curandState* localRandomState)
{
	for (int i = n - 1; i> 0; i--)
	{
		int target = static_cast<int>(round(i * curand_uniform(localRandomState)));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
}

__device__ float perlin::perlinInterpolate(vec3 c[2][2][2], float u, float v, float w)
{
	float uu = u * u * (3.f - 2.f * u);
	float vv = v * v * (3.f - 2.f * v);
	float ww = w * w * (3.f - 2.f * w);
	float acc = 0.0f;

	for (int i = 0; i < 2; i++) 
	{
		for (int j = 0; j < 2; j++) 
		{
			for (int k = 0; k < 2; k++)
			{
				vec3 weight_v(u - i, v - j, w - k);
				acc += (i * uu + (1 - i) * (1 - uu)) *
					(j * vv + (1 - j) * (1 - vv)) *
					(k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
			}
		}
	}

	return acc;
}