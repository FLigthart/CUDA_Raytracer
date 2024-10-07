#pragma once

#include <math.h>
#include "../structs/color4.h"
#include "../structs/transform.h"
#include "../structs/hitInformation.h"
class Ray;

class Shape
{

public:

	Transform transform;
	color4 color;

	__device__ virtual bool checkIntersection(Ray& ray, HitInformation& hitInformation) const = 0;
};

class ShapeList : public Shape
{
public:
	__device__ ShapeList() = default;
	__device__ ShapeList(Shape** l, int n) { list = l; listSize = n;  }

	Shape** list;
	int listSize;

	__device__ virtual bool checkIntersection(Ray& ray, HitInformation& hitInformation) const override;
};

__device__ bool inline ShapeList::checkIntersection(Ray& ray, HitInformation& hitInformation) const
{
	HitInformation temp_info;
	bool hitAnything = false;

	for (int i = 0; i < listSize; i++)
	{
		 if (list[i]->checkIntersection(ray, temp_info))
		 {
			 hitAnything = true;
			 hitInformation = temp_info;
		 }
	}

	return hitAnything;
}