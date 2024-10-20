#pragma once

#ifndef RANDOMSPHERESSCENE_H
#define RANDOMSPHERESSCENE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"

class Shape;
class Camera;

class randomSpheresScene
{
public:

	static void CreateScene(Shape** d_shapeList, Shape** d_world, Camera** d_camera, int pX, int pY, curandState* randomState);

	__host__ __device__ static int GetObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 22 * 22 + 1 + 3;
};

#endif