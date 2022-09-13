#include "cuda_spot_light.cuh"

namespace RayZath::Cuda
{
	class World;

	__host__ SpotLight::SpotLight()
		: m_size(1.0f)
		, m_angle(1.0f)
		, m_cos_angle(0.5f)
		, m_emission(0.0f)
	{}

	__host__ void SpotLight::reconstruct(
		[[maybe_unused]] const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hSpotLight->stateRegister().IsModified()) return;

		m_position = hSpotLight->position();
		m_direction = hSpotLight->direction();
		m_size = hSpotLight->size();
		m_angle = hSpotLight->GetBeamAngle();
		m_cos_angle = cosf(m_angle);
		m_color = hSpotLight->color();
		m_emission = hSpotLight->emission();

		hSpotLight->stateRegister().MakeUnmodified();
	}
}