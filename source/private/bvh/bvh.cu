#include "../../public/bvh/bvh.h"
#include <random>
#include <algorithm>
#include "../../public/mathOperations.h"

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

__device__ bvhNode::bvhNode(Shape**& shapes, int start, int end)
{
	int axis = mathOperations::randomInt(0, 2);

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
		std::sort(shapes[0] + start, shapes[0] + end, comparator);

		int middle = start + objectSpan / 2;

		left = new bvhNode(shapes, start, middle);
		right = new bvhNode(shapes, middle, end);
	}

	bbox = aabb(left->boundingBox(), right->boundingBox());
}

__device__ bool bvhNode::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const
{
	if (!bbox.checkIntersection(ray, hitRange))
		return false;

	bool hitLeft = left->checkIntersection(ray, hitRange, hitInformation);
	bool hitRight = right->checkIntersection(ray, hitRange, hitInformation);

	return hitRight || hitLeft;
}
