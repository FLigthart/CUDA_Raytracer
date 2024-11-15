#pragma once

#ifndef BVH_H
#define BVH_H

#include "../shapes/shape.h"
#include "aabb.h"

struct bvhDataNode
{
	Shape* obj;
	aabb bbox;
};

struct bvhNode : bvhDataNode
{
	int left = -1;
	int right = -1;

	bvhNode() = default;

	__host__ bvhNode(Shape* object, aabb box)
	{
		obj = object;
		bbox = box;
	}

	__device__ bvhNode(Shape* object)
	{
		obj = object;
		bbox = object->boundingBox();
	}

	__device__ static void prefillNodes(bvhNode* nodes, Shape** shapes, int listSize);

	__device__ static bool checkIntersection(const bvhNode* nodes, Ray& ray, interval hitRange, HitInformation& hitInformation);

	__host__ static int buildTree(bvhNode* nodes, int size);

	__host__ static void allocateTree(bvhNode* nodes, int size);
};

#endif
