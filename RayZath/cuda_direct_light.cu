#include "cuda_direct_light.cuh"

#include "direct_light.hpp"

namespace RayZath::Cuda
{
	__host__ DirectLight::DirectLight()
		: m_angular_size(0.2f)
		, m_cos_angular_size(0.2f)
		, m_emission(0.0f)
	{}

	__host__ void DirectLight::reconstruct(
		[[maybe_unused]] const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::DirectLight>& hDirectLight,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hDirectLight->stateRegister().IsModified()) return;

		m_direction = hDirectLight->direction();
		m_angular_size = hDirectLight->angularSize();
		m_cos_angular_size = cosf(m_angular_size);
		m_color = hDirectLight->color();
		m_emission = hDirectLight->emission();

		hDirectLight->stateRegister().MakeUnmodified();
	}
}