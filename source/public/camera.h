#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "structs/vec2.h"
#include "structs/vec3.h"

enum AAMethod : std::uint8_t
{ None, MSAA10, MSAA20, MSAA50, MSAA100, MSAA1000 };

class Camera
{

public:

    Camera() = default;
    __host__ __device__ Camera(const vec3& position, const vec3& up, const vec2& direction, float _fov, int pX, int pY, AAMethod _aaMethod, float _focusDistance, float _aperture)
    {
	    _lookFrom = position;
    	_lookUp = up;
        setLookDirection(direction);

        screenX = pX;
        screenY = pY;

        fov = validateFOV(_fov); // Vertical fov

        aaMethod = _aaMethod;

        focusDistance = _focusDistance;
        lensRadius = _aperture / 2.0f;

        initialize(); // Calculate other member variables at the end of the constructor.
    }

    int screenX;
    int screenY;

    AAMethod aaMethod;

    float lensRadius;
    float focusDistance;


    __device__ Ray makeRay(float u, float v, curandState* randomState) const;

    __host__ __device__ void initialize();  // Recalculate variables if a connect variable has changed.

    __host__ __device__ vec3 lookFrom() const { return _lookFrom; }
    __host__ __device__ vec3 lookAt() const { return _lookDirection; }
    __host__ __device__ vec3 lookUp() const { return _lookUp; }

    __host__ __device__ void setLookFrom(vec3 lookFrom) { _lookFrom = lookFrom; }

    __host__ __device__ void setLookDirection(vec2 direction);

    __host__ __device__ void setLookUp(vec3 lookUp) { _lookUp = lookUp.normalized(); }


    __host__ __device__ float getAspectRatio() const
    {
        return static_cast<float>(screenX) / static_cast<float>(screenY);
    }

    __host__ __device__ float getScreenWidth() const
    {
        return screenHeight * getAspectRatio();
    }

    __host__ __device__ vec3 getRightVector() const
    {
        return vec3(cross(_lookUp, _lookDirection).normalized());
    }

    __host__ __device__ float getFOV() const { return fov; } 
    __host__ __device__ void setFOV(float _fov)
    {
        fov = validateFOV(_fov);
        initialize();
    }


private:

    float fov;

    vec3 _lookFrom;
    vec3 _lookUp;
    vec3 _lookDirection;

    float pitch;
    float yaw;

    vec3 lowerLeftCorner;

    float screenHeight;

    vec3 horizontal;
    vec3 vertical;

    __host__ __device__ static float validateFOV(float _fov) // Checks if FOV value is valid and adjusts if it is not. FOV must be >= 10.0f && <= 180.0f.
    {
        if (_fov < 10.0f) _fov = 10.0f;
        else if (_fov > 150.0f) _fov = 150.0f;
        return _fov;
    }
};

#endif //CAMERA_H
