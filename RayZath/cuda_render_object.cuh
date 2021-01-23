#ifndef CUDA_RENDER_OBJECT_H
#define CUDA_RENDER_OBJECT_H

#include "render_object.h"
#include "exist_flag.cuh"
#include "cuda_render_parts.cuh"
#include "cuda_material.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaRenderObject : public WithExistFlag
		{
		public:
			vec3f position;
			vec3f rotation;
			vec3f center;
			vec3f scale;

			CudaBoundingBox bounding_box;

		public:
			CudaRenderObject();
			~CudaRenderObject();
		};
	}
}

#endif // !CUDA_RENDER_OBJECT_H