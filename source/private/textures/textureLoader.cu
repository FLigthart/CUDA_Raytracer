#include "../../public/textures/textureLoader.h"

#include <iostream>
#include <ostream>
#include "../../public/textures/imageTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/stb_image.h"

__host__ unsigned char* textureLoader::loadImage(const char* filePath, int& width, int& height, int& channels)
{
	unsigned char* data = stbi_load(filePath, &width, &height, &channels, 0);
	if (data == nullptr)
	{
		std::cerr << "ERROR: Could not load image " << filePath << std::endl;
	}

	return data;
}
