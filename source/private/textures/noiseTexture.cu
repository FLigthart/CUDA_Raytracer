#include "../../public/textures/noiseTexture.h"

color4 noiseTexture::value(float u, float v, const vec3& point) const
{
	vec3 s = scale * point;
	vec3 value = vec3(1.0f, 1.0f, 1.0f) * 0.5f * (1 + sin(s.z() + 10 * noise.turb(s)));
}
