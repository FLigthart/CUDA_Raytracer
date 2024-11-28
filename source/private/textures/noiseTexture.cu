#include "../../public/textures/noiseTexture.h"

__device__ color4 noiseTexture::value(float u, float v, const vec3& point) const
{
	vec3 s = scale * point;
	vec3 value = vec3(1.0f, 1.0f, 1.0f) * 0.5f * (1 + sin(s.z() + 10 * noise.turb(s)));

	return color4(value.x(), value.y(), value.z(), 1.0f);
}
