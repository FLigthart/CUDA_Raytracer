#pragma once

#include <cmath>

#include "../mathOperations.h"
#include "../util.h"
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

	__device__ virtual bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const = 0;

	__device__ virtual aabb boundingBox() const = 0;

	__device__ static vec3 getFaceNormal(const Ray& ray, const vec3& outwardNormal)
	{
		bool frontFace = dot(ray.direction(), outwardNormal) < 0;
		return frontFace ? outwardNormal : -outwardNormal;
	}
};

class translate : public Shape
{
public:
	__device__ translate(Shape* object, const vec3& offset)
		: object(object), offset(offset)
	{
		bbox = object->boundingBox() + offset;
	}
 
	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const override
	{
		Ray offsetRay(ray.origin() - offset, ray.direction(), ray.time());

		if (!object->checkIntersection(offsetRay, hitRange, hitInformation, localRandomState))
			return false;

		hitInformation.position += offset;

		return true;
	}

	__device__ aabb boundingBox() const override { return bbox; }

private:
	Shape* object;
	vec3 offset;
	aabb bbox;
};

class rotateY : public Shape
{
public:

	__device__ rotateY(Shape* object, float angle) : object(object)
	{
		float radians = mathOperations::degreesToRadians(angle);
		sinTheta = sin(radians);
		cosTheta = cos(radians);
		bbox = object->boundingBox();

		vec3 min(INFINITY, INFINITY, INFINITY);
		vec3 max(-INFINITY, -INFINITY, -INFINITY);

		for (int i = 0; i < 2; i++) 
		{
			for (int j = 0; j < 2; j++) 
			{
				for (int k = 0; k < 2; k++)
				{
					float x = i * bbox.x.max + (1 - i) * bbox.x.min;
					float y = j * bbox.y.max + (1 - j) * bbox.y.min;
					float z = k * bbox.z.max + (1 - k) * bbox.z.min;

					float newX = cosTheta * x + sinTheta * z;
					float newZ = -sinTheta * x + cosTheta * z;

					vec3 tester(newX, y, newZ);

					for (int c = 0; c < 3; c++) 
					{
						min[c] = std::fmin(min[c], tester[c]);
						max[c] = std::fmax(max[c], tester[c]);
					}
				}
			}
		}

		bbox = aabb(min, max);
	}

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const override
	{
		vec3 origin = vec3(
			(cosTheta * ray.origin().x()) - (sinTheta * ray.origin().z()),
			ray.origin().y(),
			(sinTheta * ray.origin().x()) + (cosTheta * ray.origin().z())
		);

		vec3 direction = vec3(
			(cosTheta * ray.direction().x()) - (sinTheta * ray.direction().z()),
			ray.direction().y(),
			(sinTheta * ray.direction().x()) + (cosTheta * ray.direction().z())
		);

		Ray rotatedRay(origin, direction, ray.time());

		if (!object->checkIntersection(rotatedRay, hitRange, hitInformation, localRandomState))
			return false;

		hitInformation.position = vec3(
			(cosTheta * hitInformation.position.x()) + (sinTheta * hitInformation.position.z()),
			hitInformation.position.y(),
			(-sinTheta * hitInformation.position.x()) + (cosTheta * hitInformation.position.z())
		);

		hitInformation.normal = vec3(
			(cosTheta * hitInformation.normal.x()) + (sinTheta * hitInformation.normal.z()),
			hitInformation.normal.y(),
			(-sinTheta * hitInformation.normal.x()) + (cosTheta * hitInformation.normal.z())
		);

		return true;
	}

	__device__ aabb boundingBox() const override { return bbox; }

private:
	Shape* object;
	float sinTheta;
	float cosTheta;
	aabb bbox;
};
