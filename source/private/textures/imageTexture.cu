#include "../../public/textures/imageTexture.h"

#include "../../public/structs/interval.h"
#include "../../public/textures/textureLoader.h"

#define ERROR_TEXTURE_PATH "../CUDA_Raytracer/source/assets/textureErrorImage.png"

// filePath needs to be relative from the exe location
__host__ imageTexture::imageTexture(const char* filePath)
{
	pixelData = textureLoader::loadImage(filePath, width, height, channels);

	if (pixelData == nullptr)
	{
		pixelData = textureLoader::loadImage(ERROR_TEXTURE_PATH, width, height, channels);
	}

	pixelDataSize = width * height * channels * sizeof(unsigned char);
}

__device__ imageTexture::imageTexture(unsigned char* data, int w, int h, int c)
{
	pixelData = data;
	width = w;
	height = h;
	channels = c;
	pixelDataSize = width * height * channels * sizeof(unsigned char);
}

__host__ __device__ imageTexture::~imageTexture()
{
	if (pixelData)
	{
		delete[] pixelData;
		pixelData = nullptr;
	}
}

__device__ color4 imageTexture::value(float u, float v, const vec3& point) const
{
	u = interval(0, 1).clamps(u);
	v = 1.0f - interval(0, 1).clamps(v);

	int i = static_cast<int>(u * width);
	int j = static_cast<int>(v * height);

	if (i >= width)
	{
		i = width - 1;
	}

	if (j >= height)
	{
		j = height - 1;
	}

	constexpr auto colorScale = 1.0f / 255.0f;
	auto pixel = pixelData + j * width * channels + i * channels;

	return color4(colorScale * pixel[0], colorScale * pixel[1], colorScale * pixel[2], 1.0f);
}
