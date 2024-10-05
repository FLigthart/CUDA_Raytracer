#ifndef RAY_H
#define RAY_H

#include "./structs/vec3.h"

class Ray
{
public:
	Ray() = default;
    __device__ Ray(const vec3& origin, const vec3& position) { A = origin; B = position; }

    __device__ vec3 origin() const { return A; }
    __device__ vec3 direction() const { return B; }
    __device__ vec3 pointAtMagnitude(float t) const { return A + t * B; }

    vec3 A;
    vec3 B;
};

#endif //RAY_H

