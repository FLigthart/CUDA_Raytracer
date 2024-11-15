#include "../../public/scenes/checkeredSphereScene.h"

#include "../../public/camera.h"
#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/shapes/sphere.h"
#include "../../public/textures/checkerTexture.h"
#include "../../public/structs/vec2.h"


__global__ void initializeCheckeredScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount, curandState* randomState)
{
	checkerTexture* checker = new checkerTexture(0.32f, color4(0.2f, 0.3f, 0.1f, 1.0f), color4(0.9f, 0.9f, 0.9f, 1.0f));

	d_shapeList[0] = new Sphere(vec3(0.f, -10.f, 0.f), 10.f, new lambertian(checker));
	d_shapeList[1] = new Sphere(vec3(0.f, 10.f, 0.f), 10.f, new lambertian(checker));

	bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

	*d_camera = Camera(vec3(13.f, 1.f, 0.f), vec3(0.f, 1.f, 0.f), vec2(-5.f, 180.0f),
		20.0f, pX, pY, AAMethod::MSAA100, 10.0f, 0.05f);
}

void checkeredSphereScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, curandState* localCurandState, int& listSize, int& treeSize)
{
	INIT_LIST_AND_TREE(objectCount);

	initializeCheckeredScene<<<1, 1 >>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount, localCurandState);
}
