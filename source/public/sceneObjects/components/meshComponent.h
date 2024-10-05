#pragma once
#include "../../structs/color4.h"
#include "../../shapes/shape.h"

class MeshComponent
{

public:
	color4 color;
	Shape* shape;

	__host__ __device__ MeshComponent() { color = color4::black();
		shape = nullptr; }
	__host__ __device__ MeshComponent(color4 _color, Shape* _shape) { color = _color; shape = _shape; }

	__host__ __device__ inline void SetShape(Shape* _shape)
	{
		shape = _shape;
	}

	~MeshComponent() { delete shape; }
};