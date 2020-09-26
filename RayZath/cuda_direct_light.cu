#include "cuda_direct_light.cuh"

namespace RayZath
{
	__host__ CudaDirectLight::CudaDirectLight()
	{
	}
	__host__ CudaDirectLight::~CudaDirectLight()
	{
	}

	__host__ void CudaDirectLight::Reconstruct(
		DirectLight& hDirectLight, 
		cudaStream_t& mirror_stream)
	{
		if (!hDirectLight.GetStateRegister().IsModified()) return;

		direction = hDirectLight.GetDirection();
		color = hDirectLight.GetColor();
		emission = hDirectLight.GetEmission();
		angular_size = hDirectLight.GetAngularSize();

		cos_angular_size = cosf(angular_size);

		hDirectLight.GetStateRegister().MakeUnmodified();
	}
}