#include "cuda_point_light.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		CudaPointLight::CudaPointLight()
			: size(0.1f)
		{}
		CudaPointLight::~CudaPointLight()
		{}

		void CudaPointLight::Reconstruct(
			const CudaWorld& hCudaWorld, 
			const Handle<PointLight>& hPointLight, 
			cudaStream_t& mirror_stream)
		{
			if (!hPointLight->GetStateRegister().IsModified()) return;

			position = hPointLight->GetPosition();
			size = hPointLight->GetSize();

			material.SetColor(hPointLight->GetColor());
			material.SetEmittance(hPointLight->GetEmission());

			hPointLight->GetStateRegister().MakeUnmodified();
		}
	}
}