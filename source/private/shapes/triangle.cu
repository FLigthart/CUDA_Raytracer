#include "../../public/shapes/triangle.h"

#include "../../public/util.h"

__device__ bool triangle::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const
{
	float distance = dot(a - ray.origin(), normal) / dot(ray.direction(), normal);

	if (distance < 0)
		return false;

	vec3 intersectionPoint = ray.origin() + ray.direction() * distance;

	vec3 v1 = cross(b - a, intersectionPoint - a);
	vec3 v2 = cross(a - c, intersectionPoint - c);
	vec3 v3 = cross(c - b, intersectionPoint - b);

	float f1 = dot(v1, normal);
	float f2 = dot(v2, normal);
	float f3 = dot(v3, normal);

	if (f1 >= VERY_SMALL_NUMBER && f2 >= VERY_SMALL_NUMBER && f3 >= VERY_SMALL_NUMBER)
	{
		hitInformation.position = intersectionPoint;
		hitInformation.distance = distance;
		hitInformation.normal = normal;
		hitInformation.mat = mat;

		// Calculate UV Coordinates
		vec3 vabc = cross(c - b, a - b);
		//v2
		//v3

		float labc = vabc.length() / 2.0f;
		float l2 = v2.length() / 2.0f;
		float l3 = v3.length() / 2.0f;

		float alpha = l2 / labc;
		float beta = l3 / labc;
		float gamma = 1 - alpha - beta;

		hitInformation.u = alpha * uva.x() + beta * uvb.x() + gamma * uvc.x();
		hitInformation.v = alpha * uva.y() + beta * uvb.y() + gamma * uvc.y();

		return true;
	}

	return false;
}

__device__ void triangle::setBoundingBox()
{
	float minX = minimum(a.x(), minimum(b.x(), c.x()));
	float minY = minimum(a.y(), minimum(b.y(), c.y()));
	float minZ = minimum(a.z(), minimum(b.z(), c.z()));
	vec3 minPoint = vec3(minX, minY, minZ);

	float maxX = maximum(a.x(), maximum(b.x(), c.x()));
	float maxY = maximum(a.y(), maximum(b.y(), c.y()));
	float maxZ = maximum(a.z(), maximum(b.z(), c.z()));
	vec3 maxPoint = vec3(maxX, maxY, maxZ);

	bbox = aabb(minPoint, maxPoint);
}