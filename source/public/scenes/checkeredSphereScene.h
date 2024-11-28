#pragma once

#ifndef CHECKEREDSPHERESCENE_H
#define CHECKEREDSPHERESCENE_H

#include <curand_kernel.h>

struct bvhNode;
class Shape;
class Camera;

class checkeredSphereScene
{
public:
	__host__ static void createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize);


	__host__ __device__ static int getObjectCount()
	{
		return objectCount;
	}

private:
	static constexpr int objectCount = 2;
};

#endif