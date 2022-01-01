#include "cuda_point_light.cuh"

namespace RayZath::Cuda
{
	PointLight::PointLight()
		: size(0.1f)
	{}

	void PointLight::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::PointLight>& hPointLight,
		cudaStream_t& mirror_stream)
	{
		if (!hPointLight->GetStateRegister().IsModified()) return;

		position = hPointLight->GetPosition();
		size = hPointLight->GetSize();

		material.SetColor(hPointLight->GetColor());
		material.SetEmission(hPointLight->GetEmission());

		hPointLight->GetStateRegister().MakeUnmodified();
	}
}