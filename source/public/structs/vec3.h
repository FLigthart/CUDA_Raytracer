#pragma once

#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include "curand_kernel.h"

// https://github.com/rogerallen/raytracinginoneweekendincuda/blob/ch03_rays_cuda/vec3.h

struct vec3
{
public:

    float e[3];

    vec3() = default;
    __host__ __device__ vec3(const float x, const float y, const float z) { e[0] = x; e[1] = y; e[2] = z; }

    // Getters
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float squaredLength() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __host__ __device__ inline void normalize();
    __host__ __device__ inline vec3 normalized() const;


    /*
    *  Constructors for convenience
    */

    __host__ __device__ static vec3 zero() { return vec3(0.0f, 0.0f, 0.0f); }
    __host__ __device__ static vec3 one() { return vec3(1.0f, 1.0f, 1.0f); }
};


/*
    IO Stream operations
 */

inline std::istream& operator>>(std::istream& is, vec3& t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

/*
    Scalar operations on vector
 */

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}


/*
    Vector and vector operations
 */

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v2) {
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v2) {
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    e[2] *= v2.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v2) {
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    e[2] /= v2.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v2) {
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    e[2] -= v2.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

/*
   Other Operations
*/

__host__ __device__ inline void vec3::normalize()
{
    *this = *this / length();
}

__host__ __device__ inline vec3 vec3::normalized() const
{
    return *this / length();
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

/*
 *  Random vec3's
 */

 // Returns a random vec3 between 0 and 1
__device__ static vec3 random(curandState* randomState)
{
    return vec3(curand_uniform(randomState), curand_uniform(randomState), curand_uniform(randomState));
}

__device__ static vec3 randomInUnitSphere(curandState* randomState)
{
    while (true)
    {
        // Multiply by 2 and subtract one to make the vec3 range from -1 to 1 instead of 0 to 1
        vec3 randomVec = 2.0f * random(randomState) - vec3::one();

        float vecLengthSq = randomVec.squaredLength();

        /* the squared length of the vector must be greater than a very small value.
         * Otherwise, we can get an underflow to zero on the vector. Which produces an infinite vector.
		 */
        if (vecLengthSq > 1e-160 && vecLengthSq < 1.0)
            return randomVec;
    }
}

__device__ static vec3 reflect(const vec3& inVector, const vec3& normalVector)
{
    return inVector - 2 * dot(inVector, normalVector) * normalVector;
}

__device__ static bool refract(const vec3& v, const vec3& normal, float etaiOverEtat, vec3& refracted)
{
    vec3 uv = v.normalized();
    float dt = dot(uv, normal);
    float discriminant = 1.0f - etaiOverEtat * etaiOverEtat * (1.0f - dt * dt);

    if (discriminant  > 0.0f)
    {
        refracted = etaiOverEtat * (uv - normal * dt) - normal * sqrt(discriminant);
        return true;
    }

    return false;
}

#endif //VEC3_H
