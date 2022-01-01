#ifndef CUDA_RENDER_OBJECT_H
#define CUDA_RENDER_OBJECT_H

#include "render_object.h"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"

namespace RayZath::Cuda
{
	class World;

	class RenderObject
	{
	public:
		Transformation transformation;
		BoundingBox bounding_box;
	};

}

#endif // !CUDA_RENDER_OBJECT_H