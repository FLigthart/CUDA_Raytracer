#pragma once

#ifndef RANDOMSPHERESSCENE_H
#define RANDOMSPHERESSCENE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"

struct bvhNode;
class Shape;
class Camera;

class randomSpheresScene
{
public:

	static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, curandState* localCurandState, int& listSize, int& treeSize);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 22 * 22 + 1 + 3;
};

#endif