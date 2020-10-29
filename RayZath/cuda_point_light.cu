#include "cuda_point_light.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		CudaPointLight::CudaPointLight()
			: size(0.1f)
			, emission(100.0f)
		{}
		CudaPointLight::~CudaPointLight()
		{}

		void CudaPointLight::Reconstruct(PointLight& hPointLight, cudaStream_t& mirror_stream)
		{
			if (!hPointLight.GetStateRegister().IsModified()) return;

			position = hPointLight.GetPosition();
			color = hPointLight.GetColor();
			size = hPointLight.GetSize();
			emission = hPointLight.GetEmission();

			hPointLight.GetStateRegister().MakeUnmodified();
		}
	}
}