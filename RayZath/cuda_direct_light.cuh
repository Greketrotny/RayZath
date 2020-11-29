#ifndef CUDA_DIRECT_LIGHT_CUH
#define CUDA_DIRECT_LIGHT_CUH

#include "direct_light.h"
#include "cuda_render_parts.cuh"
#include "exist_flag.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaDirectLight : public WithExistFlag
		{
		public:
			cudaVec3<float> direction;
			float angular_size;
			float cos_angular_size;
			CudaMaterial material;


		public:
			__host__ CudaDirectLight();
			__host__ ~CudaDirectLight();


		public:
			__host__ void Reconstruct(
				DirectLight& hDirectLight,
				cudaStream_t& mirror_stream);
		};
	}
}

#endif // !CUDA_DIRECT_LIGHT_CUH