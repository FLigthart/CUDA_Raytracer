#include <iostream>
#include <fstream>
#include <list>
#include <string>

#include "../public/bvh/bvh.h"
#include "../public/lights/diffuseLight.h"
#include "../public/scenes/basicSphereScenes.h"
#include "../public/scenes/checkeredSphereScene.h"
#include "../public/scenes/cornellBoxScene.h"
#include "../public/scenes/cornellSmokeScene.h"
#include "../public/scenes/perlinSphereScene.h"
#include "../public/scenes/quadsScene.h"
#include "../public/scenes/randomSpheresScene.h"
#include "../public/scenes/simpleLightScene.h"

using namespace std;

#include <cuda_runtime_api.h>
#include <vector>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

#include "../public/structs/vec3.h"
#include "../public/ray.h"
#include "../public/camera.h"
#include "../public/shapes/sphere.h"
#include "../public/structs/color4.h"
#include "../public/util.h"
#include "../public/materials/material.h"

//WARNING: Generate Relocatable Device Code = Yes in CUDA C++ Settings. Otherwise, camera.h and camera.cu won't compile. Might cause weird behaviour.

__global__ void randomInitialize(curandState* randomState)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        curand_init(2024, 0, 0, randomState);
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

__device__ color4 colorPerSample(Ray& r, bvhNode* world, Camera* camera, curandState* localRandomState)
{
    Ray currentRay = r;
    color4 currentAttenuation = color4::white();   // This decreases for each bounce. Making each bounce have less impact.
    color4 finalColor = color4::black();

    for (int i = 0; i < 50; i++) // For each pixel, cast 50 random rays.
    {
        HitInformation hitInformation;

        // Hit nothing. Return background color.
        if (!bvhNode::checkIntersection(world, currentRay, interval(0.001f, INFINITY), hitInformation, localRandomState))
        {
            finalColor += camera->background * currentAttenuation;
            break;
        }

        Ray scattered;
        color4 attenuation; // attenuation for each object the ray bounces to.

        if (!hitInformation.mat->scatter(currentRay, hitInformation, attenuation, localRandomState, scattered))
        {
            color4 colorFromEmission = hitInformation.mat->emitted(hitInformation.u, hitInformation.v, hitInformation.position);
            finalColor += colorFromEmission * currentAttenuation;
            break;
        }

        currentAttenuation *= attenuation;
        currentRay = scattered;
    }

    return finalColor;
}

__device__ vec3 colorForPixel(bvhNode* world, Camera* camera, int pixelStartX, int pixelStartY, curandState* localRandomState, int sampleSize)
{
    color4 color = color4(0.0f, 0.0f, 0.0f, 1.0f);


    for (int i = 0; i < sampleSize; i++) // For each path tracing sample
    {
        float u = static_cast<float>(pixelStartX + curand_uniform(localRandomState)) / static_cast<float>(camera->screenX);
        float v = static_cast<float>(pixelStartY + curand_uniform(localRandomState)) / static_cast<float>(camera->screenY);

        Ray r = camera->makeRay(u, v, localRandomState);

        color += colorPerSample(r, world, camera, localRandomState);
    }

    vec3 rgbValues = color.getRGB() /= static_cast<float>(sampleSize);

    // Perform square root on r, g and b
    rgbValues.e[0] = sqrt(rgbValues.x());
    rgbValues.e[1] = sqrt(rgbValues.y());
    rgbValues.e[2] = sqrt(rgbValues.z());

    return rgbValues;
}

__global__ void render(vec3* fb, Camera* camera, bvhNode* world, curandState* randomState)
{
    int pixelStartX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelStartY = threadIdx.y + blockIdx.y * blockDim.y;

    if ((pixelStartX >= camera->screenX) || (pixelStartY >= camera->screenY)) return;   // Pixels that will be rendered are out of screen.

    int pixelIndex = pixelStartY * camera->screenX + pixelStartX; // Index of pixel in array.

    curandState localRandomState = randomState[pixelIndex];

    int sampleSize; // Amount of AA samples

    switch (camera->aaMethod)
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

	    case MSAA10000:
	        sampleSize = 10000;
	        break;

	    case MSAA100000:
	        sampleSize = 100000;
	        break;

	    default: // No AA/Non-added methods
            sampleSize = 1;
    }

    fb[pixelIndex] = colorForPixel(world, camera, pixelStartX, pixelStartY, &localRandomState, sampleSize);

    randomState[pixelIndex] = localRandomState; // Make sure randomState is still set properly
}

__global__ void freeWorld(Shape** shapeList, Camera* camera)
{
	for (int i = 0; i < 2; i++)
	{
        delete shapeList[i];
	}

    delete camera;
}

__host__ std::string getScenesString(const std::vector<std::string>& worlds)
{
    std::string output;
	for (int i = 0; i < static_cast<signed>(worlds.size()); i++)
	{
        output += worlds[i] + " [" + std::to_string(i + 1) + ']';

        if (i == static_cast<signed>(worlds.size()) - 2) // On the second to last world, combine using "or" instead of ",". 
        {
            output += " or ";
        }
        else if (i < static_cast<signed>(worlds.size()) - 2)
        {
            output += ", ";
        }
	}

    output += ".";

    return output;
}

__host__ int inputTillValid(const std::string& errorMessage, const std::vector<std::string>& validInputs)
{
    string selectedWorld;
    std::getline(std::cin, selectedWorld);

    for (int i = 1; static_cast<unsigned>(i) <= validInputs.size(); i++)
    {
        if (selectedWorld == std::to_string(i))
        {
            return i;
        }
    }

    // No match. Error message and ask again for input.
    cout << errorMessage;
    return inputTillValid(errorMessage, validInputs);
}

__host__ int askUserForWorldType(const std::vector<std::string>& worlds)
{
    std::cout << "Please select a scene to render. Enter the number in front of the scene that you would like to render.\n" << "You can choose out of " << getScenesString(worlds) << '\n'; \

    std::string errorMessage = "ERROR: Please enter one of the scene names. You can choose out of " + getScenesString(worlds) + '\n';

    return inputTillValid(errorMessage, worlds);
}

__host__ int askUserForSeed()
{
    std::cout << "Enter a seed to be used for randomization (integer)" << '\n';
    string potentialSeed;
	std::cin >> potentialSeed;

    try 
    {
        return std::stoi(potentialSeed);
    }
    catch (...)
    {
        std::cerr << "ERROR: Invalid Seed Number.\n";
        return askUserForSeed();
    }
}

int main()
{
    int pX = 1920;
    int pY = 1080;

    // Divide threads into blocks to be sent to the gpu.
    int threadX = 16;
    int threadY = 16;

    vec3* fb;

    int pixelCount = pX * pY;
    size_t memorySize = pixelCount * sizeof(vec3); // One vec3 (rgb) per pixel.
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&fb), memorySize));

    // Allocate random states

    // Random state for render
    curandState* d_randomState;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_randomState), pixelCount * sizeof(curandState)));

    // Random state for world initialization
    curandState* d_randomState2;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_randomState2), sizeof(curandState)));

    // Initialize random state for world initialization
    randomInitialize<<<1, 1 >>>(d_randomState2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Different scenes the user can choose out of.
    std::vector<std::string> worlds = { "Basic Spheres", "Random Spheres", "Checkered Spheres", "Perlin Spheres", "Quads", "Simple Light", "Cornell Box", "Cornell Smoke"};

    int worldTypeIndex = askUserForWorldType(worlds);
    std::cout << worlds[worldTypeIndex - 1] << " selected.\n";

    int randomSeed = askUserForSeed();

    Shape** d_shapeList;
    int listSize, treeSize;

    bvhNode *d_bhvTree, *h_bhvTree;

    Camera* d_camera;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_camera), sizeof(Camera)));

    switch (worldTypeIndex)
	{
    case 1:
        basicSphereScene::createScene(d_shapeList, h_bhvTree, d_bhvTree, 
            d_camera, pX, pY, listSize, treeSize);
        break;

    case 2:
        randomSpheresScene::createScene(d_shapeList, h_bhvTree, d_bhvTree, 
            d_camera, pX, pY, randomSeed, listSize, treeSize);
        break;

	case 3:
		checkeredSphereScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
        d_camera, pX, pY, listSize, treeSize);
    	break;

    case 4:
        perlinSphereScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
            d_camera, pX, pY, randomSeed, listSize, treeSize);
        break;

    case 5:
        quadsScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
            d_camera, pX, pY, listSize, treeSize);
	    break;

    case 6:
        simpleLightScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
            d_camera, pX, pY, listSize, treeSize, randomSeed);
        break;

    case 7:
        cornellBoxScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
            d_camera, pX, pY, listSize, treeSize, randomSeed);
        break;

	case 8:
        cornellSmokeScene::createScene(d_shapeList, h_bhvTree, d_bhvTree,
            d_camera, pX, pY, listSize, treeSize, randomSeed);
        break;

    default:
        exit(1);
    }

    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_bhvTree, d_bhvTree, listSize * sizeof(bvhNode), cudaMemcpyDeviceToHost));

    int treeHeight = bvhNode::buildTree(h_bhvTree, listSize);
    checkCudaErrors(cudaMemcpy(d_bhvTree, h_bhvTree, treeSize * sizeof(bvhNode), cudaMemcpyHostToDevice));

    // Render a buffer
    dim3 blocks(pX / threadX + 1, pY / threadY + 1); //Block of one warp size.
    dim3 threads(threadX, threadY); // A block of amount of threads per block.

    // Ensure synchronization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Rendering a " << pX << " x " << pY << " image " << "in " << threadX << " x " << threadY << " blocks.\n";

    renderInitialize<<<blocks, threads>>>(pX, pY, d_randomState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Start counting ms here.
    clock_t start = clock();

    render<<<blocks, threads>>>(fb, d_camera, d_bhvTree, d_randomState);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t stop = clock();
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
    freeWorld<<<1, 1>>>(d_shapeList, d_camera);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_shapeList));
    checkCudaErrors(cudaFree(d_bhvTree));
    checkCudaErrors(cudaFree(d_randomState2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
