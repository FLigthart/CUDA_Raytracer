#pragma once

#ifndef QUADSSCENE_H
#define QUADSSCENE_H

#include <curand_kernel.h>

#include "cuda_runtime.h"

class Camera;
struct bvhNode;
class Shape;

class quadsScene
{
public:
	static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 5;
};
#endif
