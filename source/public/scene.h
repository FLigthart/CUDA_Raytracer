#pragma once

#include <unordered_map>
#include "shapes/shape.h"
#include "camera.h"
#include "sceneObjects/sceneObject.h"
#include "exceptionChecker.h"

class Scene
{

public:

	Camera* camera;

	VolumeObject** objectList;
	int objectListSize = 0;

	explicit Scene() { camera = new Camera(); }

	~Scene()
	{
		for (int i = 0; i < objectListSize; ++i)
		{
			checkCudaErrors(cudaFree(objectList[i]));
		}

		checkCudaErrors(cudaFree(objectList));
		checkCudaErrors(cudaFree(camera));
	}
};