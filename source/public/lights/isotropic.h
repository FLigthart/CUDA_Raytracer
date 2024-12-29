#pragma once

#ifndef ISOTROPIC_H
#define ISOTROPIC_H

#include "../materials/material.h"
#include "../structs/hitInformation.h"
#include "../textures/solidColor.h"
#include "../textures/texture.h"
#include "../ray.h"

class isotropic : public material
{
public:
    __device__ isotropic(const color4& albedo) : tex(new solidColor(albedo)) {}
    __device__ isotropic(texture* tex) : tex(tex) {}

    __device__ bool scatter(const Ray& rayIn, const HitInformation& hitInformation, color4& attenuation, curandState* randomState, Ray& scattered) const override
	{
        scattered = Ray(hitInformation.position, random(randomState), rayIn.time());
        attenuation = tex->value(hitInformation.u, hitInformation.v, hitInformation.position);

        return true;
    }
private:
	texture* tex;
};

#endif