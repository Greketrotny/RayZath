#include "cuda_spot_light.cuh"

namespace RayZath
{
	__host__ CudaSpotLight::CudaSpotLight()
		: size(1.0f)
		, emission(100.0f)
		, angle(1.0f)
		, cos_angle(0.5f)
	{}
	__host__ CudaSpotLight::~CudaSpotLight()
	{}


	__host__ void CudaSpotLight::Reconstruct(const SpotLight& hSpotLight, cudaStream_t& mirror_stream)
	{
		position = hSpotLight.GetPosition();
		direction = hSpotLight.GetDirection();
		color = hSpotLight.GetColor();
		size = hSpotLight.GetSize();
		emission = hSpotLight.GetEmission();
		angle = hSpotLight.GetBeamAngle();

		cos_angle = cos(angle);
	}
}