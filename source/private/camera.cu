#include "../public/camera.h"
#include "../public/mathOperations.h"

__device__ Ray Camera::makeRay(float u, float v, curandState* randomState)
{
    vec3 random = lensRadius * randomInUnitDisk(randomState);
    vec3 offset = getRightVector() * random.x() + _lookUp * random.y();

    vec3 direction = lowerLeftCorner + horizontal * u + v * vertical - _lookFrom - offset;
    vec3 origin = _lookFrom + offset;
    return Ray(origin, direction);
}

__host__ __device__ void Camera::initialize()
{
    // modified from https://stackoverflow.com/questions/39155572/how-to-create-an-exponential-equation-that-calculates-camera-field-of-view-again
    float fovRadians = mathOperations::degreesToRadians(getFOV());
    float halfHeight = tan(fovRadians / 2.0f);

    screenHeight = 2.0f * halfHeight;

    horizontal = getRightVector() * getScreenWidth() * focusDistance;
    vertical = _lookUp * screenHeight * focusDistance;

    vec3 screenMiddle = _lookFrom + focusDistance * _lookDirection;
    lowerLeftCorner = screenMiddle - (focusDistance * getRightVector() * (getScreenWidth() / 2.0f)) - (focusDistance * _lookUp * halfHeight);
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
