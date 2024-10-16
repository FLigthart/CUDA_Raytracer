#pragma once

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
#include "../public/materials/dielectric.h"
#include "../public/materials/material.h"
#include "../public/materials/metal.h"

//WARNING: Generate Relocatable Device Code = Yes in CUDA C++ Settings. Otherwise camera.h and camera.cu won't compile. Might cause weird behaviour.

__device__ color4 calculateBackgroundColor(const Ray& r)
{
    vec3 normalizedDirection = r.direction().normalized();
    float t = 0.5f * (normalizedDirection.y() + 1.0f);
    vec3 rgb = (1.0f - t) * vec3::one() + t * vec3(0.5f, 0.7f, 1.0f);
    return color4(rgb.x(), rgb.y(), rgb.z(), 1.0f);
}

__global__ void createWorld(Shape** d_shapeList, Shape** d_world, Camera** d_camera, int pX, int pY)
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

        *d_world = new ShapeList(d_shapeList, 6);

        //*d_camera = new Camera(vec3(-2.0f, 1.5f, -10.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-5.0f, 80.0f), 45.0f, 2.0f, pX, pY, AAMethod::MSAA1000); // standard camera
        *d_camera = new Camera(vec3(1.0f, 0.5f, -5.0f), vec3(0.0f, 1.0f, 0.0f), vec2(0.0f, 90.0f), 45.0f, 2.0f, pX, pY, AAMethod::MSAA1000); // Glass front camera
        //*d_camera = new Camera(vec3(1.0f, 3.0f, -2.0f), vec3(0.0f, 1.0f, 0.0f), vec2(-30.0f, 90.0f), 45.0f, 2.0f, pX, pY, AAMethod::MSAA1000); // Glass from top
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

__device__ color4 colorPerSample(Ray& r, Shape** world, curandState* localRandomState)
{
    Ray currentRay = r;
    color4 currentAttenuation = color4::white();   // This decreases for each bounce. Making each bounce have less impact.

    for (int i = 0; i < 50; i++) // For each pixel, cast 50 random rays.
    {
        HitInformation hitInformation;

        if ((*world)->checkIntersection(currentRay, interval(0.001f, INFINITY), hitInformation))
        {
            Ray scattered;
            color4 attenuation; // attenuation for each object the ray bounces to.

            if (hitInformation.mat->scatter(currentRay, hitInformation, attenuation, localRandomState, scattered))
            {
                currentAttenuation *= attenuation;
                currentRay = scattered;
            }
            else
            {
                return color4::black();
            }
            
            //return 0.5f * color4(hitInformation.normal.x() + 1.0f, hitInformation.normal.y() + 1.0f, hitInformation.normal.z() + 1.0f, 1.0f); //Display normals color
        }
        else
        {
            color4 color = currentAttenuation * calculateBackgroundColor(currentRay);
            return color;
        }
    }

    return color4::black(); // 0 0 0 1 values
}

__device__ vec3 colorForPixel(Shape** world, Camera** camera, int pixelStartX, int pixelStartY, curandState* localRandomState, int sampleSize)
{
    color4 color = color4(0.0f, 0.0f, 0.0f, 1.0f);


    for (int i = 0; i < sampleSize; i++) // For each path tracing sample
    {
        float u = static_cast<float>(pixelStartX + curand_uniform(localRandomState)) / static_cast<float>((*camera)->screenX);
        float v = static_cast<float>(pixelStartY + curand_uniform(localRandomState)) / static_cast<float>((*camera)->screenY);

        Ray r = (*camera)->makeRay(u, v);

        color += colorPerSample(r, world, localRandomState);
    }

    vec3 rgbValues = color.getRGB() /= static_cast<float>(sampleSize);

    // Perform square root on r, g and b
    rgbValues.e[0] = sqrt(rgbValues.x());
    rgbValues.e[1] = sqrt(rgbValues.y());
    rgbValues.e[2] = sqrt(rgbValues.z());

    return rgbValues;
}

__global__ void render(vec3* fb, Camera** camera, Shape** world, curandState* randomState)
{
    int pixelStartX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelStartY = threadIdx.y + blockIdx.y * blockDim.y;

    if ((pixelStartX >= (*camera)->screenX) || (pixelStartY >= (*camera)->screenY)) return;   // Pixels that will be rendered are out of screen.

    int pixelIndex = pixelStartY * (*camera)->screenX + pixelStartX; // Index of pixel in array.

    curandState localRandomState = randomState[pixelIndex];

    int sampleSize; // Amount of AA samples

    switch ((*camera)->aaMethod)
    {
	    case MSAA10:
            sampleSize = 4;
            break;

	    case MSAA20:
            sampleSize = 20;
            break;

        case MSAA50:
            sampleSize = 50;
            break;

	    case MSAA100:
            sampleSize = 100;
            break;

	    case MSAA1000:
	        sampleSize = 1000;
	        break;

	    default: // No AA/Non-added methods
            sampleSize = 1;
    }

    fb[pixelIndex] = colorForPixel(world, camera, pixelStartX, pixelStartY, &localRandomState, sampleSize);

    randomState[pixelIndex] = localRandomState; // Make sure randomState is still set properly
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
    int pX = 1920;
    int pY = 1080;

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

    int shapeListSize = 8;
    Shape** d_shapeList;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_shapeList), shapeListSize * sizeof(Shape*)));

    Shape** d_world;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_world), sizeof(Shape*)));

    createWorld<<<1, 1>>>(d_shapeList, d_world, d_camera, pX, pY);

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

    // Start counting ms here.
    clock_t start, stop;
    start = clock();

    render<<<blocks, threads>>>(fb, d_camera, d_world, d_randomState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = static_cast<double>(stop - start);

    std::cerr << "Image rendered in " << timer_seconds << " ms.\n";

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
