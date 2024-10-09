#include "../public/camera.h"

__device__ Ray Camera::makeRay(float u, float v)
{
    return Ray(position(), lowerLeftCorner + u * vec3(getScreenHorizontal().y(), 0.0f, 0.0f) + v * screenVertical);
}

__host__ __device__ void Camera::initialize()
{
    lowerLeftCorner = vec3(0.0f, 0.0f, focalLength) - screenVertical / 2.f - getScreenHorizontal() / 2.f;
}
