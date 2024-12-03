#include "../../public/scenes/randomSpheresScene.h"

#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "../../public/shapes/sphere.h"
#include "../../public/materials/dielectric.h"
#include "../../public/materials/metal.h"
#include "../../public/camera.h"
#include "../../public/bvh/bvh.h"
#include "../../public/util.h"

__global__ void initializeRandomScene(Shape** d_shapeList, bvhNode* d_bvhTree, Camera* d_camera, int pX, int pY, int objectCount, int randomSeed)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        INIT_RAND_LOCAL();

        material* groundMaterial = new lambertian(color4(0.5f, 0.5f, 0.5f, 1.0f));

        d_shapeList[0] = new Sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f, groundMaterial);

        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float chosenMaterial = RND;
                vec3 center(static_cast<float>(a) + RND * 0.5f, 0.2f, static_cast<float>(b) + RND * 0.5f);
                if (chosenMaterial < 0.8f)
                {
                    vec3 centerTwo = center + vec3(0.0f, 0.25f * curand_uniform(&localRandomState), 0.0f);
                    d_shapeList[i++] = new Sphere(center, centerTwo, 0.2f,
                        new lambertian(color4(RND * RND, RND * RND, RND * RND, 1.0f)));
                }
                else if (chosenMaterial < 0.95f)
                {
                    d_shapeList[i++] = new Sphere(center, 0.2f,
                        new metal(color4(0.5f * (1.0f * RND), 0.5f * (1.0f * RND), 0.5f * (1.0f * RND), 1.0f), 0.5f * RND));
                }
                else
                {
                    d_shapeList[i++] = new Sphere(center, 0.2f, new dielectric(1.5f));
                }
            }
        }

        d_shapeList[i++] = new Sphere(vec3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));

        d_shapeList[i++] = new Sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(color4(0.4f, 0.2f, 0.1f, 1.0f)));
        
        d_shapeList[i++] = new Sphere(vec3(4.0f, 1.0f, 0.0f), 1.0f, new metal(color4(0.7f, 0.6f, 0.5f, 1.0f), 0.0));

        bvhNode::prefillNodes(d_bvhTree, d_shapeList, objectCount);

        *d_camera = Camera(vec3(13.0f, 1.5f, -6.0f), vec3(0.0f, 1.0f, 0.0f),
            vec2(-12.0f, 155.0f), 30.0f, pX, pY, AAMethod::MSAA100,
            10.0f, 0.05f, color4::standardBackground());
    }
}
void randomSpheresScene::createScene(Shape**& d_shapeList, bvhNode*& h_bvhTree, bvhNode*& d_bvhTree, Camera*& d_camera, int pX, int pY, int randomSeed, int& listSize, int& treeSize)
{
    INIT_LIST_AND_TREE(objectCount);

    initializeRandomScene<<<1, 1>>>(d_shapeList, d_bvhTree, d_camera, pX, pY, objectCount, randomSeed);
}