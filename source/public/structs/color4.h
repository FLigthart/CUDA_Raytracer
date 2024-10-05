#pragma once

#include <iostream>

#include "vec3.h"

struct color4
{

public:

	float e[4];

	color4() = default;
	__host__ __device__ color4(const float r, const float g, const float b, const float a) { e[0] = r; e[1] = g; e[2] = b; e[3] = a; }

	// Getters
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }
	__host__ __device__ inline float a() const { return e[3]; }

	// Setters
	__host__ __device__ inline float& r() { return e[0]; }
	__host__ __device__ inline float& g() { return e[1]; }
	__host__ __device__ inline float& b() { return e[2]; }
	__host__ __device__ inline float& a() { return e[3]; }

    // Math Operations
    __host__ __device__ inline const color4& operator+() const { return *this; }
    __host__ __device__ inline color4 operator-() const { return color4(-e[0], -e[1], -e[2], -e[3]); }
    __host__ __device__ inline float operator[](const int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](const int i) { return e[i]; };

	__host__ __device__ inline color4& operator+=(const color4& v2);
	__host__ __device__ inline color4& operator-=(const color4& v2);
	__host__ __device__ inline color4& operator*=(const color4& v2);
	__host__ __device__ inline color4& operator/=(const color4& v2);
	__host__ __device__ inline color4& operator*=(const float t);
	__host__ __device__ inline color4& operator/=(const float t);

    // Other Functions
	__host__ __device__ vec3 getRGB() const;

    __host__ __device__ static color4 black() { return color4(0.0f, 0.0f, 0.0f, 1.0f); }
    __host__ __device__ static color4 white() { return color4(1.0f, 1.0f, 1.0f, 1.0f); }
    __host__ __device__ static color4 red() { return color4(1.0f, 0.0f, 0.0f, 1.0f); }
    __host__ __device__ static color4 green() { return color4(0.0f, 1.0f, 0.0f, 1.0f); }
    __host__ __device__ static color4 blue() { return color4(0.0f, 0.0f, 1.0f, 1.0f); }
};

/*
	IO Stream operations
 */

inline std::istream& operator>>(std::istream& is, color4& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2] >> t.e[3];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const color4& t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2] << " " << t.e[3];
	return os;
}

/*
	Scalar operations on color
 */

__host__ __device__ inline color4 operator*(const float t, const color4& v) {
	return color4(t * v.e[0], t * v.e[1], t * v.e[2], t * v.e[3]);
}

__host__ __device__ inline color4 operator/(const color4& v, const float t) {
	return color4(v.e[0] / t, v.e[1] / t, v.e[2] / t, v.e[3] / t);
}

__host__ __device__ inline color4 operator*(const color4& v, const float t) {
	return color4(t * v.e[0], t * v.e[1], t * v.e[2], t * v.e[3]);
}

/*
	Color4 and color4 operations
*/

__host__ __device__ inline color4 operator+(const color4& v1, const color4& v2) {
    return color4(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2], v1.e[3] + v2.e[3]);
}

__host__ __device__ inline color4 operator-(const color4& v1, const color4& v2) {
    return color4(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2], v1.e[3] - v2.e[4]);
}

__host__ __device__ inline color4 operator*(const color4& v1, const color4& v2) {
    return color4(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2], v1.e[3] * v2.e[3]);
}

__host__ __device__ inline color4 operator/(const color4& v1, const color4& v2) {
    return color4(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2], v1.e[3] / v2.e[3]);
}

__host__ __device__ inline color4& color4::operator+=(const color4& v2) {
    e[0] += v2.e[0];
    e[1] += v2.e[1];
    e[2] += v2.e[2];
    e[3] += v2.e[3];
    return *this;
}

__host__ __device__ inline color4& color4::operator*=(const color4& v2) {
    e[0] *= v2.e[0];
    e[1] *= v2.e[1];
    e[2] *= v2.e[2];
    e[3] *= v2.e[3];
    return *this;
}

__host__ __device__ inline color4& color4::operator/=(const color4& v2) {
    e[0] /= v2.e[0];
    e[1] /= v2.e[1];
    e[2] /= v2.e[2];
    e[3] /= v2.e[3];
    return *this;
}

__host__ __device__ inline color4& color4::operator-=(const color4& v2) {
    e[0] -= v2.e[0];
    e[1] -= v2.e[1];
    e[2] -= v2.e[2];
    e[3] -= v2.e[3];
    return *this;
}

__host__ __device__ inline color4& color4::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    e[3] *= t;
    return *this;
}

__host__ __device__ inline color4& color4::operator/=(const float t) {
    float k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    e[3] *= k;
    return *this;
}

/*
 *  Other Functions
 */

__host__ __device__ inline vec3 color4::getRGB() const
{
    return vec3(e[0], e[1], e[2]);
}