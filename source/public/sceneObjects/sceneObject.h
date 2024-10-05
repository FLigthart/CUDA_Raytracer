#pragma once

#include "../structs/transform.h"
#include "components/meshComponent.h"

// Everything that is present in the scene
class SceneObject
{

public:

	Transform transform = Transform();

	SceneObject() = default;
	virtual ~SceneObject() = default;
};

class VolumeObject : public SceneObject
{

public:

	MeshComponent* meshComponent;

	__host__ __device__ VolumeObject() { meshComponent = new MeshComponent; }



	~VolumeObject() override { delete meshComponent; }

	VolumeObject(VolumeObject& o): SceneObject(o)
	{
		meshComponent = o.meshComponent;
	}

	VolumeObject& operator=(const VolumeObject& o)
	{
		if (this != &o)
		{
			transform = o.transform;
			meshComponent = o.meshComponent;
		}

		return *this;
	}

	VolumeObject (VolumeObject&& o)
	{
		meshComponent = o.meshComponent;
		o.meshComponent = nullptr;
	}

	VolumeObject& operator=(VolumeObject&& o)
	{
		if (this != &o)
		{
			delete meshComponent;
			meshComponent = o.meshComponent;
			o.meshComponent = nullptr;
		}

		return *this;
	}
};