#ifndef CAMERA_H
#define CAMERA_H

#include "structs/vec3.h"

class Camera
{

public:

    Camera() = default;
    __host__ __device__ Camera(const vec3& position, const vec3& up, const vec3& direction, float screenHeight, float focalLength, float _fov, int pX, int pY)
    {
	    _position = position;
    	_up = up;
    	_direction = direction;

        sX = pX;
        sY = pY;

        float aspectRatio = static_cast<float>(pX) / static_cast<float>(pY);
        float screenWidth = screenHeight * aspectRatio;

        lowerLeftCorner = vec3(-screenWidth / 2.f, -screenHeight / 2.f, focalLength);
        horizontal = vec3(screenWidth, 0.0f, 0.0f);
        vertical = vec3(0.0f, screenHeight, 0.0f);

        fov = _fov;
    }

    __host__ __device__ vec3 position() const { return _position; }
    __host__ __device__ vec3 direction() const { return _direction; }
    __host__ __device__ vec3 up() const { return _up; }

    vec3 _position;
    vec3 _up;
    vec3 _direction;

    vec3 lowerLeftCorner;
    vec3 horizontal;
    vec3 vertical;

    int sX;
    int sY;

    float fov;

    __device__ Ray makeRay(float u, float v);
};

__device__ inline Ray Camera::makeRay(float u, float v)
{
    return Ray(position(), lowerLeftCorner + u * horizontal + v * vertical);
}

#endif //CAMERA_H
