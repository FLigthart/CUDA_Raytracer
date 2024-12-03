#include "../../public/scenes/basicSphereScenes.h"

#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "../../public/shapes/sphere.h"
#include "../../public/materials/dielectric.h"
#include "../../public/materials/metal.h"
#include "../../public/camera.h"
#include "../../public/bvh/bvh.h"
#include "../../public/util.h"
#include "../../public/textures/checkerTexture.h"

__global__ void initializeBasicSpheres(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto checker = new checkerTexture(0.32f, color4(0.2f, 0.3f, 0.1f, 1.0f), color4(0.9f, 0.9f, 0.9f, 1.0f));

        d_shapeList[0] = new Sphere(vec3(0.0f, 1.0f, 2.0f), 1.2f, new lambertian(color4(0.1f, 0.2f, 0.5f, 1.0f)));

        d_shapeList[1] = new Sphere(vec3(0.0f, -20.0f, 5.0f), 20.0f, new lambertian(checker));

        d_shapeList[2] = new Sphere(vec3(2.0f, 0.5f, 2.0f), 0.75f, new metal(color4(0.69f, 0.55f, 0.34f, 1.0f), 0.8f));

        d_shapeList[3] = new Sphere(vec3(-2.0f, 0.5f, 2.0f), 0.75f, new metal(color4(0.8f, 0.6f, 0.2f, 1.0f), 0.1f));

        /*
         * Hollow glass sphere (glass sphere with glass refractionIndex and air sphere)
         */
        d_shapeList[4] = new Sphere(vec3(1.0f, 0.5f, 0.5f), 0.5f, new dielectric(1.50f));

        d_shapeList[5] = new Sphere(vec3(1.0f, 0.5f, 0.5f), 0.40f, new dielectric(1.00f / 1.50f));

        bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

        *d_camera = Camera(vec3(0.0f, 1.5f, -3.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-5.0f, 90.0f),
            45.0f, pX, pY, AAMethod::MSAA100, 5.0f, 0.0f, color4::standardBackground()); // standard camera
    }
}


void basicSphereScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int& listSize, int& treeSize)
{
    INIT_LIST_AND_TREE(objectCount);

    initializeBasicSpheres<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount);
}