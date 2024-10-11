#pragma once

#ifndef VEC2_H
#define VEC2_H

#include <cmath>
#include <iostream>

struct vec2
{
public:
	float e[2];

	vec2() = default;
	__host__ __device__ vec2(const float x, const float y) { e[0] = x; e[1] = y;}

	// Getters
	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }

	__host__ __device__ inline const vec2& operator+() const { return *this; }
	__host__ __device__ inline vec2 operator-() const { return vec2(-e[0], -e[1]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; };

	__host__ __device__ inline vec2& operator+=(const vec2& v2);
	__host__ __device__ inline vec2& operator-=(const vec2& v2);
	__host__ __device__ inline vec2& operator*=(const vec2& v2);
	__host__ __device__ inline vec2& operator/=(const vec2& v2);
	__host__ __device__ inline vec2& operator*=(const float t);
	__host__ __device__ inline vec2& operator/=(const float t);

	__host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1]); }
	__host__ __device__ inline float squaredLength() const { return e[0] * e[0] + e[1] * e[1]; }

	__host__ __device__ inline void normalize();
	__host__ __device__ inline vec2 normalized() const;

	__host__ __device__ static vec2 zero() { return vec2(0.0f, 0.0f); }
	__host__ __device__ static vec2 one() { return vec2(1.0f, 1.0f); }
};

/*
	IO Stream operations
 */

inline std::istream& operator>>(std::istream& is, vec2& t)
{
	is >> t.e[0] >> t.e[1];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec2& t) {
	os << t.e[0] << " " << t.e[1];
	return os;
}


/*
	Scalar operations on vector
 */

__host__ __device__ inline vec2 operator*(float t, const vec2& v) {
	return vec2(t * v.e[0], t * v.e[1]);
}

__host__ __device__ inline vec2 operator/(const vec2& v, float t) {
	return vec2(v.e[0] / t, v.e[1] / t);
}

__host__ __device__ inline vec2 operator*(const vec2& v, float t) {
	return vec2(t * v.e[0], t * v.e[1]);
}

/*
    Vector and vector operations
 */

__host__ __device__ inline vec2 operator+(const vec2& v1, const vec2& v2) {
    return vec2(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1]);
}

__host__ __device__ inline vec2 operator-(const vec2& v1, const vec2& v2) {
    return vec2(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1]);
}

__host__ __device__ inline vec2 operator*(const vec2& v1, const vec2& v2) {
    return vec2(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1]);
}

__host__ __device__ inline vec2 operator/(const vec2& v1, const vec2& v2) {
    return vec2(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1]);
}

__host__ __device__ inline vec2& vec2::operator+=(const vec2& v2) {
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    return *this;
}

__host__ __device__ inline vec2& vec2::operator*=(const vec2& v2) {
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    return *this;
}

__host__ __device__ inline vec2& vec2::operator/=(const vec2& v2) {
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    return *this;
}

__host__ __device__ inline vec2& vec2::operator-=(const vec2& v2) {
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    return *this;
}

__host__ __device__ inline vec2& vec2::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    return *this;
}

__host__ __device__ inline vec2& vec2::operator/=(const float t) {
    float k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    return *this;
}

/*
   Other Operations
*/

__host__ __device__ inline void vec2::normalize()
{
    *this = *this / length();
}

__host__ __device__ inline vec2 vec2::normalized() const
{
    return *this / length();
}

#endif