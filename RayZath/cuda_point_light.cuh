#ifndef CUDA_POINT_LIGHT_H
#define CUDA_POINT_LIGHT_H

#include "point_light.h"
#include "exist_flag.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaPointLight : public WithExistFlag
		{
		public:
			cudaVec3<float> position;
			CudaColor<float> color;
			float size;
			float emission;


		public:
			__host__ CudaPointLight();
			__host__ ~CudaPointLight();


		public:
			__host__ void Reconstruct(PointLight& host_light, cudaStream_t& mirror_stream);
		};
	}
}

#endif // !CUDA_POINT_LIGHT_H