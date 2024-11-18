#pragma once

#ifndef TEXTURELOADER_H
#define TEXTURELOADER_H

#include "cuda_runtime.h"

class imageTexture;

class textureLoader
{
public:
	__host__ static unsigned char* loadImage(const char* filePath, int& width, int& height, int& channels);
};

#endif
