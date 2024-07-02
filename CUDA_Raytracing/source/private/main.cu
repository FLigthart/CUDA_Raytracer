#include <iostream>
#include <fstream>
#include <time.h>

#include "../public/vec3.h"
#include "../public/ray.h"
#include "../public/camera.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ );


void check_cuda(cudaError_t result, char const *const function, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << function << "' \n";

        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 calculateColor(const ray& r)
{
    vec3 normalizedDirection = r.direction().normalized();
    float t = 0.05f * (normalizedDirection.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void render(vec3* fb, int maxPixelX, int maxPixelY, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int pixelStartX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelStartY = threadIdx.y + blockIdx.y * blockDim.y;

    if ((pixelStartX >= maxPixelX) || (pixelStartY >= maxPixelY)) return;   // Pixels that will be rendered are out of screen.

    int pixelIndex = pixelStartY * maxPixelX + pixelStartX; // Index of pixel in array.

    // 0 to 1 r, g and b values of pixel are stored in array. Color is based on pixel position on screen.
    float u = static_cast<float>(pixelStartX) / static_cast<float>(maxPixelX);
    float v = static_cast<float>(pixelStartY) / static_cast<float>(maxPixelY);

    ray r(origin, lowerLeftCorner + u * horizontal + v * vertical);
    fb[pixelIndex] = calculateColor(r);
}

int main() {

    int pX = 1920;
    int pY = 1080;

    float aspectRatioY = static_cast<float>(1080) / static_cast<float>(1920);

    float screenHeight = 2.0f;
    float screenWidth = screenHeight / aspectRatioY;
    float focalLength = 1.0f;

    camera camera(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));

    // Devide threads into blocks to be sent to the gpu.
    int threadX = 12;
    int threadY = 12;

    std::cerr << "Rendering a " << pX << " x " << pY << " image " << "in " << threadX << " x " << threadY << " blocks.\n";

    int pixelCount = pX * pY;
    size_t memorySize = pixelCount * sizeof(vec3); // One vec3 (rgb) per pixel.

    vec3* fb;
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&fb), memorySize));

    // Render a buffer
    dim3 blocks(pX / threadX + 1, pY / threadY + 1); //Block of one warp size.
    dim3 threads(threadX, threadY); // A block of amount of threads per block.

    // Start timer
    clock_t start, stop;
    start = clock();

    render<<<blocks, threads>>>(fb, pX, pY,
        vec3(-2.0, -1.0, -1.0),
            vec3(screenWidth, 0.0, 0.0),
            vec3(0.0, screenHeight, 0.0),
            camera.position());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();

    float renderTime = static_cast<float>(stop - start) / CLOCKS_PER_SEC * 1000;
    std::cerr << "Image rendered in " << renderTime << " ms.\n";

    // Open the file
    std::ofstream ofs("image.ppm", std::ios::out | std::ios::trunc |std::ios::binary);  // Empty file before writing

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
}
