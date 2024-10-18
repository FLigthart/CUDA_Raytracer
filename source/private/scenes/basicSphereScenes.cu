#include "../../public/scenes/basicSphereScenes.h"

#include "curand_kernel.h"
#include "device_launch_parameters.h"

#include "../../public/exceptionChecker.h"
#include "../../public/shapes/sphere.h"
#include "../../public/materials/dielectric.h"
#include "../../public/materials/metal.h"
#include "../../public/camera.h"

__global__ void InitializeBasicSpheres(Shape** d_shapeList, Shape** d_world, Camera** d_camera, int pX, int pY, int objectCount)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_shapeList[0] = new Sphere(1.2f, new lambertian(color4(0.1f, 0.2f, 0.5f, 1.0f)));
        d_shapeList[0]->transform.position = vec3(0.0f, 1.0f, 2.0f);

        d_shapeList[1] = new Sphere(20.0f, new lambertian(color4(0.8f, 0.8f, 0.0f, 1.0f)));
        d_shapeList[1]->transform.position = vec3(0.0f, -20.0f, 5.0f);

        d_shapeList[2] = new Sphere(0.75f, new metal(color4(0.69f, 0.55f, 0.34f, 1.0f), 0.8f));
        d_shapeList[2]->transform.position = vec3(2.0f, 0.5f, 2.0f);

        d_shapeList[3] = new Sphere(0.75f, new metal(color4(0.8f, 0.6f, 0.2f, 1.0f), 0.1f));
        d_shapeList[3]->transform.position = vec3(-2.0f, 0.5f, 2.0f);

        /*
         * Hollow glass sphere (glass sphere with glass refractionIndex and air sphere)
         */
        d_shapeList[4] = new Sphere(0.5f, new dielectric(1.50f));
        d_shapeList[4]->transform.position = vec3(1.0f, 0.5f, 0.5f);

        d_shapeList[5] = new Sphere(0.40f, new dielectric(1.00f / 1.50f));
        d_shapeList[5]->transform.position = vec3(1.0f, 0.5f, 0.5f);

        *d_world = new ShapeList(d_shapeList, objectCount);

        *d_camera = new Camera(vec3(0.0f, 1.5f, -3.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-5.0f, 90.0f), 45.0f, pX, pY, AAMethod::MSAA1000, 5.0f, 0.0f); // standard camera
    }
}

void basicSphereScene::CreateBasicSpheres(Shape** d_shapeList, Shape** d_world, Camera** d_camera, int pX, int pY)
{
    InitializeBasicSpheres<<<1, 1>>>(d_shapeList, d_world, d_camera, pX, pY, objectCount);
}