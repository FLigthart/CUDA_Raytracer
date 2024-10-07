#include <iostream>
#include <fstream>

#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

#include "../public/structs/vec3.h"
#include "../public/ray.h"
#include "../public/camera.h"
#include "../public/shapes/sphere.h"
#include "../public/structs/color4.h"
#include "../public/exceptionChecker.h"

__device__ color4 calculateBackgroundColor(const Ray& r)
{
    vec3 normalizedDirection = r.direction().normalized();
    float t = (normalizedDirection.y() + 1.0f);
    vec3 rgb = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
    return color4(rgb.x(), rgb.y(), rgb.z(), 1.0f);
}

__global__ void createWorld(Shape** d_shapeList, Shape** d_world, Camera** d_camera, float screenHeight, float focalLength, float fov, int pX, int pY)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
        d_shapeList[0] = new Sphere(1.0f);
        d_shapeList[0]->transform.position = vec3(0.0f, 0.0f, 3.0f);

        d_shapeList[1] = new Sphere(2.0f);
        d_shapeList[1]->transform.position = vec3(2.5f, 0.0f, 4.0f);
        d_shapeList[1]->color = color4::green();

        *d_world = new ShapeList(d_shapeList, 2);

        *d_camera = new Camera(vec3(0.0f, 0.0f, -3.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f),
            screenHeight, focalLength, fov, pX, pY, AAMethod::None);
	}
}

__global__ void renderInitialize(int sX, int sY, curandState* randomState)  // Initializes random for every thread. Is used for MSAA.
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= sX) || (j >= sY)) return;

    int pixelIndex = j * sX + i;

    // Every thread gets the same seed, a different sequence number and no offset.
    curand_init(2024, pixelIndex, 0, &randomState[pixelIndex]);
}

__device__ vec3 renderNoAA(Shape** world, Camera** camera, int pixelStartX, int pixelStartY)
{
    HitInformation hitInformation;

    float u = static_cast<float>(pixelStartX) / static_cast<float>((*camera)->screenX);
    float v = static_cast<float>(pixelStartY) / static_cast<float>((*camera)->screenY);

    Ray r = (*camera)->makeRay(u, v);

    if ((*world)->checkIntersection(r, hitInformation))
    {
        return hitInformation.color.getRGB();
        //fb[pixelIndex] = 0.5f * vec3(hitInformation.normal.x() + 1.0f, hitInformation.normal.y() + 1.0f, hitInformation.normal.z() + 1.0f); //Display normals color
    }
    else
    {
        return calculateBackgroundColor(r).getRGB();
    }
}

__device__ vec3 renderAA(Shape** world, Camera** camera, int pixelStartX, int pixelStartY, curandState localRandomState, int sampleSize)
{
    color4 color = color4(0.0f, 0.0f, 0.0f, 1.0f);

    for (int i = 0; i < sampleSize; i++)
    {
        HitInformation hitInformation;

        float u = static_cast<float>(pixelStartX + curand_uniform(&localRandomState)) / static_cast<float>((*camera)->screenX);
        float v = static_cast<float>(pixelStartY + curand_uniform(&localRandomState)) / static_cast<float>((*camera)->screenY);

        Ray r = (*camera)->makeRay(u, v);

        if ((*world)->checkIntersection(r, hitInformation))
        {
            color += hitInformation.color;
            // color += 0.5f * vec3(hitInformation.normal.x() + 1.0f, hitInformation.normal.y() + 1.0f, hitInformation.normal.z() + 1.0f); //Display normals color
        }
        else
        {
            color += calculateBackgroundColor(r);
        }
    }

    return (color.getRGB() / static_cast<float>(sampleSize));
}

__global__ void render(vec3* fb, Camera** camera, Shape** world, curandState* randomState)
{
    int pixelStartX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelStartY = threadIdx.y + blockIdx.y * blockDim.y;

    if ((pixelStartX >= (*camera)->screenX) || (pixelStartY >= (*camera)->screenY)) return;   // Pixels that will be rendered are out of screen.

    int pixelIndex = pixelStartY * (*camera)->screenX + pixelStartX; // Index of pixel in array.

    curandState localRandomState = randomState[pixelIndex];

    switch ((*camera)->aaMethod)
    {
	    case MSAA4:
            fb[pixelIndex] = renderAA(world, camera, pixelStartX, pixelStartY, localRandomState, 4);
            break;

	    case MSAA8:
            fb[pixelIndex] = renderAA(world, camera, pixelStartX, pixelStartY, localRandomState, 8);
            break;

	    case MSAA16:
            fb[pixelIndex] = renderAA(world, camera, pixelStartX, pixelStartY, localRandomState, 16);
            break;

	    default: // No AA/Non-added methods
            fb[pixelIndex] = renderNoAA(world, camera, pixelStartX, pixelStartY);
    }
}

__global__ void freeWorld(Shape** shapeList, Shape** world, Camera** camera)
{
	for (int i = 0; i < 2; i++)
	{
        delete shapeList[i];
	}

    delete *world;
    delete *camera;
}

int main()
{
    int pX = 400; //1920
    int pY = 235; //1080

    float screenHeight = 2.0f;
    float focalLength = 1.0f;
    float fov = 20.0f; //No effect yet

    // Divide threads into blocks to be sent to the gpu.
    int threadX = 12;
    int threadY = 12;

    std::cerr << "Rendering a " << pX << " x " << pY << " image " << "in " << threadX << " x " << threadY << " blocks.\n";

    vec3* fb;

    int pixelCount = pX * pY;
    size_t memorySize = pixelCount * sizeof(vec3); // One vec3 (rgb) per pixel.
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb), memorySize));

    Camera** d_camera;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera*)));

    int shapeListSize = 2;
    Shape** d_shapeList;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_shapeList), shapeListSize * sizeof(Shape*)));

    Shape** d_world;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(Shape*)));

    createWorld<<<1, 1>>>(d_shapeList, d_world, d_camera, screenHeight, focalLength, fov, pX, pY);

    // Render a buffer
    dim3 blocks(pX / threadX + 1, pY / threadY + 1); //Block of one warp size.
    dim3 threads(threadX, threadY); // A block of amount of threads per block.

    // Allocate random state
    curandState* d_randomState;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_randomState), pixelCount * sizeof(curandState)));

    // Ensure synchronization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    renderInitialize<<<blocks, threads>>>(pX, pY, d_randomState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, d_camera, d_world, d_randomState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Image rendered.\n";

    // Open the file
    std::ofstream ofs("image.ppm", std::ios::out | std::ios::trunc | std::ios::binary);  // Empty file before writing


    // Output an image
    ofs << "P3\n" << pX << " " << pY << "\n255\n";
    for (int y = pY - 1; y >= 0; y--)
    {
        for (int x = 0; x < pX; x++)
        {
            size_t pixelIndex = y * pX + x;

            float r = fb[pixelIndex].x();
            float g = fb[pixelIndex].y();
            float b = fb[pixelIndex].z();

            int ir = static_cast<int>(255.99 * r);
            int ig = static_cast<int>(255.99 * g);
            int ib = static_cast<int>(255.99 * b);

            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }
    ofs.close();

    std::cerr << "Writing render to file finished.";

    checkCudaErrors(cudaDeviceSynchronize());
    freeWorld<<<1, 1 >>>(d_shapeList, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_shapeList));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_randomState));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
