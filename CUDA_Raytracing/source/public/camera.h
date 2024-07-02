#ifndef CAMERA_H
#define CAMERA_H

class vec3;

class camera
{

public:

    __host__ __device__ camera() {}
    __host__ __device__ camera(const vec3& position, const vec3& up, const vec3& direction) { _position = position; _up = up; _direction = direction; }
    __host__ __device__ vec3 position() const { return _position; }
    __host__ __device__ vec3 direction() const { return _direction; }
    __host__ __device__ vec3 up() const { return _up; }

    vec3 _position;
    vec3 _up;
    vec3 _direction;
};

#endif //CAMERA_H
