#pragma once

#include <math.h>
#include "../structs/color4.h"
#include "../structs/transform.h"

struct HitInformation;
class Ray;

class Shape
{

public:

	__device__ virtual bool checkIntersection(Ray& ray, Transform& transform, HitInformation& hitInformation) const = 0;

	/* The virtual destructor ensures that the base class is deleted when a derived class is deleted.
	 * If you hold a point to a base class, but initialize it as a derived class, it will also delete the reference to the base class.
	 * E.g. Shape* shape = new sphere(); delete shape; will also delete the pointer to the base class with a virtual destructor */
	__device__ virtual ~Shape() = default;
};