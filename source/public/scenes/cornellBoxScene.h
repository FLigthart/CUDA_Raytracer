#pragma once

#ifndef CORNELL_BOX_SCENE
#define CORNELL_BOX_SCENE

#include <curand_kernel.h>
#include "cuda_runtime.h"

struct bvhNode;
class Shape;
class Camera;

class cornellBoxScene
{
public:

	static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize, int localRandomSeed);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 6;
};

#endif
