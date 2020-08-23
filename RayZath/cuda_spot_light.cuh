#ifndef CUDA_SPOT_LIGHT_CUH
#define CUDA_SPOT_LIGHT_CUH

#include "spot_light.h"
#include "cuda_render_parts.cuh"
#include "world_object.h"
#include "exist_flag.cuh"


namespace RayZath
{
	class CudaSpotLight : public WithExistFlag
	{
	public:
		cudaVec3<float> position;
		cudaVec3<float> direction;
		CudaColor<float> color;
		float size, emission;
		float angle, cos_angle;


	public:
		__host__ CudaSpotLight();
		__host__ ~CudaSpotLight();


	public:
		__host__ void Reconstruct(const SpotLight& hSpotLight, cudaStream_t& mirror_stream);
	};
}

#endif // !CUDA_SPOT_LIGHT_CUH