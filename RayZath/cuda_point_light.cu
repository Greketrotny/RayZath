#include "cuda_point_light.cuh"

namespace RayZath::Cuda
{
	PointLight::PointLight()
		: m_size(0.1f)
		, m_emission(0.0f)
	{}

	void PointLight::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::PointLight>& hPointLight,
		cudaStream_t& mirror_stream)
	{
		if (!hPointLight->GetStateRegister().IsModified()) return;

		m_position = hPointLight->GetPosition();
		m_size = hPointLight->GetSize();
		m_color = hPointLight->GetColor();
		m_emission = hPointLight->GetEmission();

		hPointLight->GetStateRegister().MakeUnmodified();
	}
}