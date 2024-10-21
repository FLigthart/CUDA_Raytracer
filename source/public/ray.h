#pragma once

#ifndef RAY_H
#define RAY_H

#include "./structs/vec3.h"

class Ray
{
public:
	Ray() = default;
    __host__ __device__ Ray(const vec3& origin, const vec3& position) { A = origin; B = position; _time = 0.0f; }
    __host__ __device__ Ray(const vec3& origin, const vec3& position, float time) { A = origin; B = position; _time = time; }

    __host__ __device__ static Ray zero() { return Ray(vec3::zero(), vec3::zero()); }

    __host__ __device__ vec3 origin() const { return A; }
    __host__ __device__ vec3 direction() const { return B; }
    __host__ __device__ vec3 pointAtMagnitude(float t) const { return A + t * B; }
    __host__ __device__ float time() const { return _time; }

    __host__ __device__ vec3 at(float t) const
    {
        return origin() + t * direction();
    }

private:
    vec3 A;
    vec3 B;
    float _time;
};

#endif //RAY_H

