#include "../../public/scenes/checkeredSphereScene.h"

#include "../../public/camera.h"
#include "../../public/util.h"
#include "../../public/bvh/bvh.h"
#include "../../public/shapes/sphere.h"
#include "../../public/textures/checkerTexture.h"
#include "../../public/structs/vec2.h"
#include "../../public/textures/textureLoader.h"
#include "../../public/textures/imageTexture.h"


__global__ void initializeCheckeredScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount,
	unsigned char* earthTextureData, int earthWidth, int earthHeight, int earthChannels)
{
	auto checker = new checkerTexture(0.32f, color4(0.2f, 0.3f, 0.1f, 1.0f), color4(0.9f, 0.9f, 0.9f, 1.0f));

	auto earthTexture = new imageTexture(earthTextureData, earthWidth, earthHeight, earthChannels);

	d_shapeList[0] = new Sphere(vec3(0.f, -10.f, 0.f), 10.f, new lambertian(checker));
	d_shapeList[1] = new Sphere(vec3(0.f, 1.f, 0.f), 1.f, new lambertian(earthTexture));

	bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

	*d_camera = Camera(vec3(13.f, 1.f, 0.f), vec3(0.f, 1.f, 0.f), vec2(-5.f, 180.0f),
		20.0f, pX, pY, AAMethod::MSAA100, 10.0f, 0.05f, color4::standardBackground());
}

void checkeredSphereScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY,
	int& listSize, int& treeSize)
{
	INIT_LIST_AND_TREE(objectCount);

	auto h_earthTexture = imageTexture("../CUDA_Raytracer/source/assets/earthmap.jpg");
	unsigned char* d_earthTextureData;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_earthTextureData), h_earthTexture.pixelDataSize));
	checkCudaErrors(cudaMemcpy(d_earthTextureData, h_earthTexture.pixelData, h_earthTexture.pixelDataSize, cudaMemcpyHostToDevice));

	initializeCheckeredScene<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount,
		d_earthTextureData, h_earthTexture.width, h_earthTexture.height, h_earthTexture.channels);
}
