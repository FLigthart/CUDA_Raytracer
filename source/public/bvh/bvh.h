#pragma once

#ifndef BVH_H
#define BVH_H

#include "../shapes/shape.h"
#include "aabb.h"

class bvhNode : public Shape
{
public:
	__device__ bvhNode(Shape** shapeList, int objectCount) : bvhNode(shapeList, 0, objectCount) {}

	__device__ bvhNode(Shape**& shapes, int start, int end);

	__device__ static bool boxCompare(Shape* a, Shape* b, int axisIndex);

	__device__ static bool boxXCompare(Shape* a, Shape* b);
	__device__ static bool boxYCompare(Shape* a, Shape* b);
	__device__ static bool boxZCompare(Shape* a, Shape* b);

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const override;

	__device__ aabb boundingBox() const override { return bbox; }

private:
	Shape* left;
	Shape* right;
	aabb bbox;
};

#endif
