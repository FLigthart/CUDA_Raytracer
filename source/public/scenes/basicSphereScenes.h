#pragma once

#ifndef BASICSPHERESCENE_H
#define BASICSPHERESCENE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"

class bvhNode;
class Shape;
class Camera;

class basicSphereScene
{
public:

	static void CreateScene(bvhNode*& d_bvhTree, Shape** d_shapeList, Camera** d_camera, int pX, int pY, curandState* localCurandState);

	__host__ __device__ static int GetObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 6;
};

#endif