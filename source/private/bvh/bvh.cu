#include "../../public/bvh/bvh.h"
#include <random>
#include "../../public/util.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

__device__ bool bvhNode::boxCompare(Shape* a, Shape* b, int axisIndex)
{
	interval aAxisInterval = a->boundingBox().axisInterval(axisIndex);
	interval bAxisInterval = b->boundingBox().axisInterval(axisIndex);

	return aAxisInterval.min < bAxisInterval.min;
}

__device__ bool bvhNode::boxXCompare(Shape* a, Shape* b)
{
	return boxCompare(a, b, 0);
}

__device__ bool bvhNode::boxYCompare(Shape* a, Shape* b)
{
	return boxCompare(a, b, 1);
}

__device__ bool bvhNode::boxZCompare(Shape* a, Shape* b)
{
	return boxCompare(a, b, 2);
}

__device__ bvhNode::bvhNode(Shape** shapes, int start, int end)
{
	bbox = aabb::empty();
	for (int objectIndex = start; objectIndex < end; objectIndex++)
	{
		bbox = aabb(bbox, shapes[objectIndex]->boundingBox());
	}

	int axis = bbox.longestAxis();

	auto comparator = (axis == 0) ? boxXCompare
		: (axis == 1) ? boxYCompare
		: boxZCompare;

	int objectSpan = end - start;

	switch (objectSpan)
	{
	case 1:
		left = right = shapes[start];
		break;
	case 2:
		left = shapes[start];
		right = shapes[start + 1];
		break;
	default:
		thrust::sort(shapes + start, shapes + end, comparator);

		int middle = start + objectSpan / 2;

		left = new bvhNode(shapes, start, middle);
		right = new bvhNode(shapes, middle, end);
	}
}

__device__ bool bvhNode::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const
{
	if (!bbox.checkIntersection(ray, hitRange))
		return false;

	bool hitLeft = left->checkIntersection(ray, hitRange, hitInformation);
	bool hitRight = right->checkIntersection(ray, hitRange, hitInformation);

	return hitRight || hitLeft;
}

__host__ void bvhNode::initializeTree(int size, bvhNode* d_bvhTree)
{
	int treeSize = 2 * size;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_bvhTree), treeSize * sizeof(bvhNode)));
}