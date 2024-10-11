#include "../public/camera.h"
#include "../public/mathOperations.h"

__device__ Ray Camera::makeRay(float u, float v)
{
    vec3 direction = lowerLeftCorner + getRightVector() * u * getScreenWidth() + _lookUp * v * screenHeight - _lookFrom;
    return Ray(_lookFrom, direction);
}

__host__ __device__ void Camera::initialize()
{
    vec3 screenMiddle = _lookFrom + getScreenDistance() * _lookDirection;
    lowerLeftCorner = screenMiddle - (getRightVector() * getScreenWidth() / 2) - (_lookUp * screenHeight / 2);
}

__host__ __device__ void Camera::setLookDirection(vec2 direction)
{
    // Constrain pitch to prevent gimbal lock
    if (direction.x() > 89.0f)
    {
        direction = vec2(89.0f, direction.y());
    }
    else if (direction.x() < -89.0f)
    {
        direction = vec2(-89.0f, direction.y());
    }

    pitch = direction.x();
    yaw = direction.y();

    _lookDirection = vec3(cos(mathOperations::degreesToRadians(yaw)) * cos(mathOperations::degreesToRadians(pitch)), 
        sin(mathOperations::degreesToRadians(pitch)), 
        sin(mathOperations::degreesToRadians(yaw)) * cos(mathOperations::degreesToRadians(pitch)));
    _lookDirection.normalize();

    vec3 right = vec3(cross(vec3(0, 1, 0), _lookDirection)).normalized();
    _lookUp = cross(_lookDirection, right).normalized();
}

__host__ __device__ float Camera::getScreenDistance()
{
    // modified from https://stackoverflow.com/questions/39155572/how-to-create-an-exponential-equation-that-calculates-camera-field-of-view-again
    float fovRadians = mathOperations::degreesToRadians(getFOV());
    float distance = getScreenWidth() / (float)(2 * tan(fovRadians / 2));
    return distance;
}
