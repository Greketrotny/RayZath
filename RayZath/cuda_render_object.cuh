#ifndef CUDA_RENDER_OBJECT_H
#define CUDA_RENDER_OBJECT_H

#include "render_object.h"
#include "exist_flag.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	class CudaRenderObject : public WithExistFlag
	{
	public:
		cudaVec3<float> position;
		cudaVec3<float> rotation;
		CudaMaterial material;


	public:
		CudaRenderObject();
		~CudaRenderObject();
	};
}

#endif // !CUDA_RENDER_OBJECT_H