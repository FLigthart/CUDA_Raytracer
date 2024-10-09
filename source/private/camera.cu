#include "../public/camera.h"
#include "../public/mathOperations.h"

__device__ Ray Camera::makeRay(float u, float v)
{
    return Ray(_position, lowerLeftCorner + u * getScreenHorizontal() + v * screenVertical - _position);
}

__host__ __device__ void Camera::initialize()
{
    float theta = mathOperations::degreesToRadians(fov);
    float h = std::tan(theta / 2.0f);
    screenVertical = vec3(0.0f, 2 * h * focalLength, 0.0f);

    lowerLeftCorner = _position + vec3(0.0f, 0.0f, focalLength) - screenVertical / 2.f - getScreenHorizontal() / 2.f;
}
