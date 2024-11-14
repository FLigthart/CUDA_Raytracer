#pragma once

#include <cmath>

#include "../materials/lambertian.h"
#include "../structs/ShapeTransform.h"
#include "../structs/hitInformation.h"
#include "../structs/interval.h"
#include "../bvh/aabb.h"

class Ray;

class Shape
{

public:

	ShapeTransform transform;

	__device__ virtual bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const = 0;

	__device__ virtual aabb boundingBox() const = 0;
};

class ShapeList : public Shape
{
public:
	__device__ ShapeList(Shape** l, int n);

	Shape** list;
	int listSize;

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const override;

	__device__ aabb boundingBox() const override;

private:
	aabb bbox;
};

__device__ inline ShapeList::ShapeList(Shape** l, int n) : list(l), listSize(n)
{
	for (int i = 0; i < n; i++) // For each object, expand the bbox so that it contains the new shape.
	{
		bbox = aabb(bbox, list[i]->boundingBox());
	}
}

__device__ bool inline ShapeList::checkIntersection(Ray& ray, const interval hitRange, HitInformation& hitInformation) const
{
	HitInformation temp_info;
	bool hitAnything = false;
	float shortestDistance = INFINITY;

	for (int i = 0; i < listSize; i++)
	{
		 if (list[i]->checkIntersection(ray, hitRange, temp_info) && temp_info.distance < shortestDistance)
		 {
			 hitAnything = true;
			 shortestDistance = temp_info.distance;
			 hitInformation = temp_info;
		 }
	}

	return hitAnything;
}

__device__ inline aabb ShapeList::boundingBox() const
{
	return bbox;
}



