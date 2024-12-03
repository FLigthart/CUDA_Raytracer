#pragma once

#ifndef SIMPLE_LIGHT_SCENE_H
#define SIMPLE_LIGHT_SCENE_H

#include <curand_kernel.h>
#include "cuda_runtime.h"

struct bvhNode;
class Shape;
class Camera;

class simpleLightScene
{
public:

	static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize, int localRandomSeed);

	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 4;
};

#endif