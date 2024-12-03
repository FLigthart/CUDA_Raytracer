#include "../../public/scenes/quadsScene.h"
#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/camera.h"
#include "../../public/shapes/quad.h"
#include "../../public/shapes/triangle.h"

__global__ void initializeQuadsScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount)
{
	lambertian* red = new lambertian(color4(1.0f, 0.2f, 0.2f, 1.0f));
	lambertian* green = new lambertian(color4(0.2f, 1.f, 0.2f, 1.0f));
	lambertian* blue = new lambertian(color4(0.2f, 0.2f, 1.0f, 1.0f));
	lambertian* orange = new lambertian(color4(1.0f, 0.5f, 0.0f, 1.0f));
	lambertian* teal = new lambertian(color4(0.2f, 0.8f, 0.8f, 1.0f));

	d_shapeList[0] = new quad(vec3(-3.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, -4.0f), vec3(0.0f, 4.0f, 0.0f), red);

	// Back Triangle
	d_shapeList[1] = new triangle(vec3(-2.0f, -2.0f, 8.0f), vec3(2.0f, -2.0f, 8.0f), vec3(2.0f, 2.0f, 8.0f), green);

	d_shapeList[2] = new quad(vec3(3.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, 4.0f), vec3(0.0f, 4.0f, 0.0f), blue);
	d_shapeList[3] = new quad(vec3(0.0f, 3.0f, 1.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 4.0f), orange);
	d_shapeList[4] = new quad(vec3(0.0f, -3.0f, 3.0f), vec3(4.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, -4.0f), teal);

	bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

	*d_camera = Camera(vec3(0.0f, 0.0f, -9.0f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 90.0f),
		45.0f, pX, pY, AAMethod::MSAA1000, 5.0f, 0.0f, color4::standardBackground());
}

void quadsScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize)
{
	INIT_LIST_AND_TREE(objectCount);

	initializeQuadsScene<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount);
}
