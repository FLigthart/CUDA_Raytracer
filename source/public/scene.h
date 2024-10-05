#pragma once

#include <unordered_map>
#include "shapes/shape.h"
#include "camera.h"

class SceneObject;

class Scene
{
private:
	std::unordered_map<unsigned int, std::shared_ptr<SceneObject>> objectList;

public:
	Camera* camera;

	explicit Scene(Camera* mainCamera) { camera = mainCamera; }

	~Scene()
	{
		delete camera;
	}
};