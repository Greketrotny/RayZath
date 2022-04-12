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

	__host__ void SpotLight::Reconstruct(
		[[maybe_unused]] const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hSpotLight->GetStateRegister().IsModified()) return;

		m_position = hSpotLight->GetPosition();
		m_direction = hSpotLight->GetDirection();
		m_size = hSpotLight->GetSize();
		m_angle = hSpotLight->GetBeamAngle();
		m_cos_angle = cosf(m_angle);
		m_color = hSpotLight->GetColor();
		m_emission = hSpotLight->GetEmission();

		hSpotLight->GetStateRegister().MakeUnmodified();
	}
}