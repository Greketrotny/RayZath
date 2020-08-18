#include "cuda_point_light.cuh"

namespace RayZath
{
	CudaPointLight::CudaPointLight()
		: size(0.1f)
		, emission(100.0f)
	{}
	CudaPointLight::~CudaPointLight()
	{}

	void CudaPointLight::Reconstruct(PointLight& hPointLight, cudaStream_t& mirror_stream)
	{
		position = hPointLight.GetPosition();
		color = hPointLight.GetColor();
		size = hPointLight.GetSize();
		emission = hPointLight.GetEmission();

		hPointLight.Updated();
	}
}