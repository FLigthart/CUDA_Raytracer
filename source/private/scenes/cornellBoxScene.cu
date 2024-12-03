#include "../../public/scenes/cornellBoxScene.h"

#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/lights/diffuseLight.h"
#include "../../public/shapes/quad.h"
#include "../../public/camera.h"
#include "../../public/shapes/sphere.h"

__global__ void initializeCornellBoxScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount, int randomSeed)
{
	lambertian* red = new lambertian(color4(0.65f, 0.05f, 0.05f, 1.0f));
	lambertian* green = new lambertian(color4(0.12f, 0.45f, 0.15f, 1.0f));
	lambertian* white = new lambertian(color4(0.73f, 0.73f, 0.73f, 1.0f));
	diffuseLight* light = new diffuseLight(color4(15.0f, 15.0f, 15.0f, 1.0f));

	d_shapeList[0] = new quad(vec3(555.0f, 277.5f, 277.5f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), green); // Right
	d_shapeList[1] = new quad(vec3(0.0f, 277.5f, 277.5f), vec3(0.0f, 555.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), red); // Left
	d_shapeList[2] = new quad(vec3(277.5f, 0.0f, 277.5f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 555.0f), white); // Bottom
	d_shapeList[3] = new quad(vec3(277.5f, 555.0f, 277.5f), vec3(-555.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -555.0f), white); //Top
	d_shapeList[4] = new quad(vec3(277.5f, 277.5f, 555.0f), vec3(555.0f, 0.0f, 0.0f), vec3(0.0f, 555.0f, 0.01f), white); //Back

	d_shapeList[5] = new quad(vec3(277.5f, 554.0f, 277.5f), vec3(-130.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -105.0f), light); // Box Light

	d_shapeList[6] = new box(vec3(130.0f, 0.0f, 65.0f), vec3(295.0f, 165.0f, 230.0f), white);
	d_shapeList[7] = new box(vec3(265.0f, 0.0f, 295.0f), vec3(430.0f, 330.0f, 460.0f), white);

	bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

	*d_camera = Camera(vec3(277.5f, 277.5f, -800.0f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 90.0f),
		40.0f, pX, pY, AAMethod::MSAA10000, 5.0f, 0.0f, color4::black());
}

void cornellBoxScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera,
	int pX, int pY, int& listSize, int& treeSize, int localRandomSeed)
{
	INIT_LIST_AND_TREE(objectCount);

	initializeCornellBoxScene<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount, localRandomSeed);
}
