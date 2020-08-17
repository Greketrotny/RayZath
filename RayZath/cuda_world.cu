#include "cuda_world.cuh"

namespace RayZath
{
	HostPinnedMemory CudaWorld::m_hpm(0xFFFF);

	CudaWorld::CudaWorld()
	{}
	CudaWorld::~CudaWorld()
	{}

	void CudaWorld::Reconstruct(
		const World& host_world,
		cudaStream_t* const mirror_stream)
	{
		cameras.Reconstruct(host_world.GetCameras(), m_hpm, mirror_stream);
		//pointLights.Reconstruct(host_world.GetPointLights(), m_host_pinned_memory, mirror_stream);
	}
}