#include "../public/scenes.h"
#include "../public/exceptionChecker.h"
#include "../public/sceneObjects/sceneObject.h"
#include "../public/shapes/sphere.h"

__global__ void k_simpleSphere(Scene* scene)
{
	scene->objectList[0] = new VolumeObject();
	scene->objectList[0]->transform.position.z() = 3.0f;
	scene->objectList[0]->meshComponent->color = color4::red();
	scene->objectList[0]->meshComponent->SetShape(new Sphere(1.0f));

	*scene->camera = Camera(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
}

// Allocate memory for kernel here and call kernel.
void simpleSphere(Scene*& scene)
{
    scene->objectListSize = 1;  // Only one object in the scene

    // Allocate memory for the Scene object on the device
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene), sizeof(Scene)));

    // Allocate memory for the object list (array of pointers) on the device
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene->objectList), scene->objectListSize * sizeof(VolumeObject*)));

    // Allocate the object on the host
    VolumeObject* h_volumeObject = new VolumeObject();

    // Initialize the VolumeObject (host-side initialization)
    h_volumeObject->transform.position.z() = 3.0f;
    h_volumeObject->meshComponent->color = color4::red();
    h_volumeObject->meshComponent->SetShape(new Sphere(1.0f));   // Make sure memory is properly handled in SetShape

    // Allocate memory for the VolumeObject on the device
    VolumeObject* d_volumeObject;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_volumeObject), sizeof(VolumeObject)));

    // Copy the VolumeObject from host to device
    checkCudaErrors(cudaMemcpy(d_volumeObject, h_volumeObject, sizeof(VolumeObject), cudaMemcpyHostToDevice));

    // Copy the device pointer to the objectList on the device
    checkCudaErrors(cudaMemcpy(&scene->objectList[0], &d_volumeObject, sizeof(VolumeObject*), cudaMemcpyHostToDevice));

    // Allocate memory for the camera on the device
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene->camera), sizeof(Camera)));

    // Now launch the kernel to work on the scene
    k_simpleSphere << <dim3(1, 1), dim3(1, 1) >> > (scene);
}