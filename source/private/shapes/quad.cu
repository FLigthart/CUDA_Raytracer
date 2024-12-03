#include "../../public/shapes/quad.h"
#include "../../public/util.h"

__device__ void quad::setBoundingBox()
{
	vec3 Q = GetQ();

	aabb boxOne = aabb(Q + u, Q + v);
	aabb boxTwo = aabb(Q, Q + u + v);
	bbox = aabb(boxOne, boxTwo);
}

__device__ vec3 quad::GetQ() const
{
	return transform.position.origin() - 0.5f * u - 0.5f * v;
}

__device__ bool quad::isInterior(float a, float b, HitInformation& hitInformation) const
{
	interval unitInterval = interval(0.f, 1.f);

	if (!unitInterval.surrounds(a) || !unitInterval.surrounds(b))
		return false;

	hitInformation.u = a;
	hitInformation.v = b;
	return true;
}

__device__ bool quad::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const
{
	vec3 Q = GetQ();
	float D = dot(normal, Q);

	float denom = dot(normal, ray.direction());

	// No hit if the ray is parallel to the plane
	if (abs(denom) < VERY_SMALL_NUMBER)
		return false;

	// No hit if the hit point parameter t is outside the ray interval
	float t = (D - dot(normal, ray.origin())) / denom;
	if (!hitRange.surrounds(t))
		return false;

	vec3 intersection = ray.at(t);
	vec3 planarHitptVector = intersection - Q;

	vec3 n = cross(u, v);
	vec3 w = n / dot(n, n);

	float alpha = dot(w, cross(planarHitptVector, v));
	float beta = dot(w, cross(u, planarHitptVector));

	if (!isInterior(alpha, beta, hitInformation))
		return false;

	hitInformation.distance = t;
	hitInformation.position = intersection;
	hitInformation.mat = mat;
	hitInformation.normal = getFaceNormal(ray, normal);

	return true;
}
