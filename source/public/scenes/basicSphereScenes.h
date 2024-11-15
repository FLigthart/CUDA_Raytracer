#pragma once

#ifndef BASICSPHERESCENE_H
#define BASICSPHERESCENE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"

struct bvhNode;
class Shape;
class Camera;

class basicSphereScene
{
public:

	static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, curandState* localCurandState, int& listSize, int& treeSize);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 6;
};

#endif