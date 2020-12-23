#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "spot_light.h"
#include "cuda_render_parts.cuh"
#include "world_object.h"
#include "exist_flag.cuh"


namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		class CudaSpotLight : public WithExistFlag
		{
		public:
			cudaVec3<float> position;
			cudaVec3<float> direction;
			float size;
			float angle, cos_angle;
			float sharpness;
			CudaMaterial material;


		public:
			__host__ CudaSpotLight();
			__host__ ~CudaSpotLight();


		public:
			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld, 
				const Handle<SpotLight>& hSpotLight, 
				cudaStream_t& mirror_stream);
		};
	}
}

#endif // !CUDA_SPOT_LIGHT_CUH