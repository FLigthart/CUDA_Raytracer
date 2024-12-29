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

__device__ bool quad::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const
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

/*
 *	Box Functions
 */

__device__ box::box(const vec3& a, const vec3& b, material* mat)
{
	vec3 min = vec3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
	vec3 max = vec3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

	vec3 dx = vec3(max.x() - min.x(), 0.1f, 0.0f);
	vec3 dy = vec3(0.0f, max.y() - min.y(), 0.1f);
	vec3 dz = vec3(0.1f, 0, max.z() - min.z());

	sides[0] = quad(vec3(min.x(), min.y(), max.z()) + dx * 0.5f + dy * 0.5f, dx, dy, mat);  // front
	sides[1] = quad(vec3(max.x(), min.y(), max.z()) - dz * 0.5f + dy * 0.5f, -dz, dy, mat); // right
	sides[2] = quad(vec3(max.x(), min.y(), min.z()) - dx * 0.5f + dy * 0.5f, -dx, dy, mat); // back
	sides[3] = quad(vec3(min.x(), min.y(), min.z()) + dz * 0.5f + dy * 0.5f, dz, dy, mat);  // left
	sides[4] = quad(vec3(min.x(), max.y(), max.z()) + dx * 0.5f - dz * 0.5f, dx, -dz, mat); // top
	sides[5] = quad(vec3(min.x(), min.y(), min.z()) + dx * 0.5f + dz * 0.5f, dx, dz, mat);  // bottom

	bbox = aabb(min, max);
}

__device__ box::~box()
{
	delete[] sides;
	delete mat;
}

__device__ bool box::checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation, curandState* localRandomState) const
{
	HitInformation tempHitInformation;
	bool hitAnything = false;
	float closestSoFar = hitRange.max;

	for (const quad& side : sides)
	{
		if (side.checkIntersection(ray, interval(hitRange.min, closestSoFar), tempHitInformation, localRandomState))
		{
			hitAnything = true;
			closestSoFar = tempHitInformation.distance;
			hitInformation = tempHitInformation;
		}
	}

	return hitAnything;
}
