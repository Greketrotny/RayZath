#include "cuda_spot_light.cuh"

namespace RayZath::Cuda
{
	class World;

	__host__ SpotLight::SpotLight()
		: size(1.0f)
		, angle(1.0f)
		, cos_angle(0.5f)
		, sharpness(1.0f)
	{}

	__host__ void SpotLight::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::SpotLight>& hSpotLight,
		cudaStream_t& mirror_stream)
	{
		if (!hSpotLight->GetStateRegister().IsModified()) return;

		position = hSpotLight->GetPosition();
		direction = hSpotLight->GetDirection();
		size = hSpotLight->GetSize();
		angle = hSpotLight->GetBeamAngle();
		cos_angle = cos(angle);
		sharpness = hSpotLight->GetSharpness();

		material.SetColor(hSpotLight->GetColor());
		material.SetEmission(hSpotLight->GetEmission());

		hSpotLight->GetStateRegister().MakeUnmodified();
	}
}