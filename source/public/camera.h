#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "structs/vec3.h"

enum AAMethod : std::uint8_t
{ None, MSAA4, MSAA8, MSAA16 };

class Camera
{

public:

    Camera() = default;
    __host__ __device__ Camera(const vec3& position, const vec3& up, const vec3& direction, float screenHeight, float _focalLength, float _fov, int pX, int pY, AAMethod _aaMethod)
    {
	    _position = position;
    	_up = up;
    	_direction = direction;

        screenX = pX;
        screenY = pY;

        focalLength = _focalLength;

        screenVertical = vec3(0.0f, screenHeight, 0.0f);

        fov = validateFOV(_fov);

        aaMethod = _aaMethod;

        initialize();
    }

    int screenX;
    int screenY;

    AAMethod aaMethod;

    __device__ Ray makeRay(float u, float v);

    __host__ __device__ void initialize();  // Recalculate variables if a connect variable has changed.



    __host__ __device__ vec3 position() const { return _position; }
    __host__ __device__ vec3 direction() const { return _direction; }
    __host__ __device__ vec3 up() const { return _up; }

    __host__ __device__ void setPosition(vec3 position) { _position = position; }
    __host__ __device__ void setDirection(vec3 direction) { _direction = direction; }
    __host__ __device__ void setUp(vec3 up) { _up = up; }


    __host__ __device__ float getAspectRatio()
    {
        return static_cast<float>(screenX) / static_cast<float>(screenY);
    }

    __host__ __device__ vec3 getScreenHorizontal()
    {
	    return vec3(screenVertical.y() * getAspectRatio(), 0.0f, 0.0f);
    }

    __host__ __device__ vec3 getRightVector()
    {
        return vec3(cross(_up, _direction).normalized());
    }

    __host__ __device__ float getFOV() { return fov; }
    __host__ __device__ void setFOV(float _fov)
    {
        fov = validateFOV(_fov);
        initialize();
    }


private:

    vec3 _position;
    vec3 _up;
    vec3 _direction;

    vec3 lowerLeftCorner;
    vec3 screenVertical;

    float focalLength;
    float fov;

    __host__ __device__ static float validateFOV(float _fov) // Checks if FOV value is valid and adjusts if it is not. FOV must be >= 10.0f && <= 180.0f.
    {
        if (_fov < 10.0f) _fov = 10.0f;
        else if (_fov > 180.0f) _fov = 180.0f;
        return _fov;
    }
};

#endif //CAMERA_H
