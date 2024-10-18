#pragma once

#ifndef  BASICSPHERESCENE_H
#define BASICSPHERESCENE_H

#include "cuda_runtime.h"

class Shape;
class Camera;

class basicSphereScene
{
public:

	static void CreateBasicSpheres(Shape** d_shapeList, Shape** d_world, Camera** d_camera, int pX, int pY);

	__host__ __device__ static int GetObjectCount()
	{
		return objectCount;
	}

private:

	static constexpr int objectCount = 6;
};

#endif