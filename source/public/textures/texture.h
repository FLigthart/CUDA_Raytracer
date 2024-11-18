#pragma once

#include "../structs/color4.h"

#ifndef TEXTURE_H
#define TEXTURE_H

struct vec3;

class texture
{
public:
	__device__ virtual color4 value(float u, float v, const vec3& point) const = 0;
};

#endif