#include "cuda_direct_light.cuh"

namespace RayZath::Cuda
{
	__host__ DirectLight::DirectLight()
		: m_angular_size(0.2f)
		, m_cos_angular_size(0.2f)
		, m_emission(0.0f)
	{}

	__host__ void DirectLight::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::DirectLight>& hDirectLight,
		cudaStream_t& mirror_stream)
	{
		if (!hDirectLight->GetStateRegister().IsModified()) return;

		m_direction = hDirectLight->GetDirection();
		m_angular_size = hDirectLight->GetAngularSize();
		m_cos_angular_size = cosf(m_angular_size);
		m_color = hDirectLight->GetColor();
		m_emission = hDirectLight->GetEmission();

		hDirectLight->GetStateRegister().MakeUnmodified();
	}
}