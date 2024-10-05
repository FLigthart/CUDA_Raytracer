#include "../public/scenes.h"
#include "../public/exceptionChecker.h"
#include "../public/shapes/sphere.h"

__global__ void k_simpleSphere(Shape** d_shapeList, Camera* d_camera)
{
	d_shapeList[0] = new Sphere(1.0f);

	*d_camera = Camera(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));

	// VolumeObject* sphereObject = new VolumeObject();
// sphereObject->transform.position.z() = 3.0f;
// sphereObject->meshComponent->color = color4::red();
// sphereObject->meshComponent->SetShape(new Sphere(1.0f));
}

// Allocate memory for kernel here and call kernel.
void simpleSphere(Shape**& d_shapeList, Camera*& d_camera)
{
	int objectListSize = 1;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_shapeList), objectListSize * sizeof(Shape)));

	k_simpleSphere<<<dim3(1, 1), dim3(1, 1)>>>(d_shapeList, d_camera);
}
