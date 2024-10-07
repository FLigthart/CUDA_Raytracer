#include <iostream>
#include <fstream>
#include <time.h>

#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include "../public/structs/vec3.h"
#include "../public/ray.h"
#include "../public/camera.h"
#include "../public/shapes/sphere.h"
#include "../public/structs/color4.h"
#include "../public/exceptionChecker.h"
#include "../public/scenes.h"

__device__ vec3 calculateBackgroundColor(const Ray& r)
{
    vec3 normalizedDirection = r.direction().normalized();
    float t = (normalizedDirection.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3* fb, int maxPixelX, int maxPixelY, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, Scene* scene)
{
    int pixelStartX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelStartY = threadIdx.y + blockIdx.y * blockDim.y;

    if ((pixelStartX >= maxPixelX) || (pixelStartY >= maxPixelY)) return;   // Pixels that will be rendered are out of screen.

    int pixelIndex = pixelStartY * maxPixelX + pixelStartX; // Index of pixel in array.

    float u = static_cast<float>(pixelStartX) / static_cast<float>(maxPixelX);
    float v = static_cast<float>(pixelStartY) / static_cast<float>(maxPixelY);

    Ray r(scene->camera->position(), lowerLeftCorner + u * horizontal + v * vertical);

    HitInformation hitInformation;

    // ERROR IS HERE (LINE 54). SEE compute-sanitizer "D:\HomeProjects\CUDA_Raytracer\x64\Debug\CUDA_Raytracer.exe". 8 bytes read attempt. SHAPE IS INVALID

    Transform transform(vec3::one());
    if (scene->objectList && scene->objectList[0]->meshComponent->shape->checkIntersection(r, transform, hitInformation))
    {
        fb[pixelIndex] = scene->objectList[0]->meshComponent->color.getRGB();
    }
    else
    {
        fb[pixelIndex] = calculateBackgroundColor(r);
    }
}

int main() {

    int pX = 1920;
    int pY = 1080;

    float aspectRatioY = static_cast<float>(1080) / static_cast<float>(1920);

    float screenHeight = 2.0f;
    float screenWidth = screenHeight / aspectRatioY;
    float focalLength = 1.0f;

    // Divide threads into blocks to be sent to the gpu.
    int threadX = 12;
    int threadY = 12;

    std::cerr << "Rendering a " << pX << " x " << pY << " image " << "in " << threadX << " x " << threadY << " blocks.\n";

    int pixelCount = pX * pY;
    size_t memorySize = pixelCount * sizeof(vec3); // One vec3 (rgb) per pixel.

    vec3* fb;
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb), memorySize));

    // Render a buffer
    dim3 blocks(pX / threadX + 1, pY / threadY + 1); //Block of one warp size.
    dim3 threads(threadX, threadY); // A block of amount of threads per block.

    Scene* d_scene = new Scene();

    simpleSphere(d_scene);

    // Ensure synchronization
    checkCudaErrors(cudaDeviceSynchronize());

    render <<<blocks, threads>>> (fb, pX, pY,
        vec3(-screenWidth / 2, -screenHeight / 2, focalLength),
        vec3(screenWidth, 0.0, 0.0),
        vec3(0.0, screenHeight, 0.0), d_scene);

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

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_scene));
}
