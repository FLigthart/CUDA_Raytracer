#pragma once

#ifndef CORNELL_SMOKE_SCENE_H
#define CORNELL_SMOKE_SCENE_H

#include <curand_kernel.h>
#include "cuda_runtime.h"

class Camera;
struct bvhNode;
class Shape;

class cornellSmokeScene
{
public:
	__host__ static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize, int localRandomSeed);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:
	static constexpr int objectCount = 8;
};

#endif