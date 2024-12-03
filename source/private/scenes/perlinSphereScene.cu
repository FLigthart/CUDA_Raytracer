#include "../../public/scenes/perlinSphereScene.h"

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/shapes/sphere.h"
#include "../../public/textures/noiseTexture.h"
#include "../../public/structs/vec2.h"
#include "../../public/camera.h"

__global__ void initializePerlinSpheres(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int randomSeed, int objectCount)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		INIT_RAND_LOCAL();

		auto perlinTexture = new noiseTexture(2.f, &localRandomState);
		auto perlinTextureBig = new noiseTexture(1.f, &localRandomState);
		d_shapeList[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(perlinTextureBig));
		d_shapeList[1] = new Sphere(vec3(3.0f, 2.f, 0.0f), 2.f, new lambertian(perlinTexture));

		bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

		*d_camera = Camera(vec3(0.0f, 1.0f, -7.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-5.0f, 70.0f),
			45.0f, pX, pY, AAMethod::MSAA100, 3.0f, 0.0f, color4::standardBackground()); // standard camera
	}
}


void perlinSphereScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int randomSeed, int& listSize, int& treeSize)
{
	INIT_LIST_AND_TREE(objectCount);

	initializePerlinSpheres<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, randomSeed, objectCount);
}
