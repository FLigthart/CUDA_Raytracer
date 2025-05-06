#include "../../public/scenes/simpleLightScene.h"

#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/lights/diffuseLight.h"
#include "../../public/shapes/sphere.h"
#include "../../public/textures/noiseTexture.h"
#include "../../public/camera.h"
#include "../../public/shapes/quad.h"

__global__ void initializeSimpleLightScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount, int randomSeed)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		INIT_RAND_LOCAL();

		auto pertext = new noiseTexture(4.0f, &localRandomState);
		d_shapeList[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(pertext));
		d_shapeList[1] = new Sphere(vec3(0.0f, 2.0f, 0.0f), 2.0f, new lambertian(pertext));

		auto diffLight = new diffuseLight(color4(4.0f, 4.0f, 4.0f, 1.0f));
		d_shapeList[2] = new quad(vec3(4.0f, 2.0f, -2.0f), vec3(2.0f, 0.0f, 0.0f), vec3(0.0f, 2.0f, 0.0f), diffLight);
		d_shapeList[3] = new Sphere(vec3(0.0f, 7.0f, 0.0f), 2.0f, diffLight);

		bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

		*d_camera = Camera(vec3(26.0f, 3.0f, 6.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-15.0f, 180.0f),
			45.0f, pX, pY, AAMethod::MSAA10000, 5.0f, 0.0f, color4::black());
	}
}
void simpleLightScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera,
                                   int pX, int pY, int& listSize, int& treeSize, int randomSeed)
{
	INIT_LIST_AND_TREE(objectCount);

	initializeSimpleLightScene<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount, randomSeed);
}
