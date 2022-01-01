#include "cuda_direct_light.cuh"

namespace RayZath::Cuda
{
	__host__ DirectLight::DirectLight()
		: angular_size(0.2f)
		, cos_angular_size(0.2f)
	{}

	__host__ void DirectLight::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::DirectLight>& hDirectLight,
		cudaStream_t& mirror_stream)
	{
		if (!hDirectLight->GetStateRegister().IsModified()) return;

		direction = hDirectLight->GetDirection();
		angular_size = hDirectLight->GetAngularSize();
		cos_angular_size = cosf(angular_size);

		material.SetColor(hDirectLight->GetColor());
		material.SetEmission(hDirectLight->GetEmission());

		hDirectLight->GetStateRegister().MakeUnmodified();
	}
}