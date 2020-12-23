#include "cuda_direct_light.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		__host__ CudaDirectLight::CudaDirectLight()
			: angular_size(0.2f)
			, cos_angular_size(0.2f)
		{}
		__host__ CudaDirectLight::~CudaDirectLight()
		{}

		__host__ void CudaDirectLight::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<DirectLight>& hDirectLight,
			cudaStream_t& mirror_stream)
		{
			if (!hDirectLight->GetStateRegister().IsModified()) return;

			direction = hDirectLight->GetDirection();
			angular_size = hDirectLight->GetAngularSize();

			material.color = hDirectLight->GetColor();
			material.emittance = hDirectLight->GetEmission();

			cos_angular_size = cosf(angular_size);

			hDirectLight->GetStateRegister().MakeUnmodified();
		}
	}
}