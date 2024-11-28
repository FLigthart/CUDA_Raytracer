#include "../../public/scenes/perlinSphereScene.h"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "../../public/util.h"
#include "../../public/bvh/bvh.h"

__global__ void initializePerlinSpheres(Shape**& d_shapeList, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int randomSeed, int objectCount)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		INIT_RAND_LOCAL();


	}
}


void perlinSphereScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int randomSeed, int& listSize, int& treeSize)
{
	INIT_LIST_AND_TREE(objectCount);

	initializePerlinSpheres<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, randomSeed, objectCount);
}
