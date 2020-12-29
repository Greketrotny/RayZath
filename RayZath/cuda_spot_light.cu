#include "cuda_spot_light.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		class CudaWorld;

		__host__ CudaSpotLight::CudaSpotLight()
			: size(1.0f)
			, angle(1.0f)
			, cos_angle(0.5f)
			, sharpness(1.0f)
		{}
		__host__ CudaSpotLight::~CudaSpotLight()
		{}


		__host__ void CudaSpotLight::Reconstruct(
			const CudaWorld& hCudaWorld, 
			const Handle<SpotLight>& hSpotLight, 
			cudaStream_t& mirror_stream)
		{
			if (!hSpotLight->GetStateRegister().IsModified()) return;

			position = hSpotLight->GetPosition();
			direction = hSpotLight->GetDirection();
			size = hSpotLight->GetSize();
			angle = hSpotLight->GetBeamAngle();
			cos_angle = cos(angle);
			sharpness = hSpotLight->GetSharpness();

			material.color = hSpotLight->GetColor();
			material.emittance = hSpotLight->GetEmission();
			material.transmittance = 0.0f;

			hSpotLight->GetStateRegister().MakeUnmodified();
		}
	}
}