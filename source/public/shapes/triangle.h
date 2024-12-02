#pragma once

#ifndef TRIANGE_H
#define TRIANGLE_H
#include "shape.h"
#include "../structs/vec2.h"

class triangle : public Shape
{
public:
	// Image Textured triangle
	__device__ triangle(const vec3& pointOne, const vec3& pointTwo, const vec3& pointThree, 
		const vec2& uv1, const vec2& uv2, const vec2& uv3, material* material)
		: a(pointOne), b(pointTwo), c(pointThree), uva(uv1), uvb(uv2), uvc(uv3), mat(material)
	{
		normal = cross(b - a, c - a).normalized();

		setBoundingBox();
	}

	// Non-image textured triangle
	__device__ triangle(const vec3& pointOne, const vec3& pointTwo, const vec3& pointThree, material* material)
		: a(pointOne), b(pointTwo), c(pointThree), mat(material)
	{
		normal = cross(b - a, c - a).normalized();

		uva = vec2::zero();
		uvb = vec2(1.0f, 0.0f);
		uvc = vec2::one();

		setBoundingBox();
	}

	__device__ bool checkIntersection(Ray& ray, interval hitRange, HitInformation& hitInformation) const override;
	__device__ aabb boundingBox() const override { return bbox; }

private:

	__device__ void setBoundingBox();

	vec3 a;
	vec3 b;
	vec3 c;

	vec2 uva;
	vec2 uvb;
	vec2 uvc;

	vec3 normal;

	material* mat;
	aabb bbox;
};

#endif