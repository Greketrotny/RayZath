#include "cuda_world.cuh"

namespace RayZath
{
	HostPinnedMemory CudaWorld::m_hpm(0xFFFF);

	CudaWorld::CudaWorld()
	{}
	CudaWorld::~CudaWorld()
	{}

	void CudaWorld::Reconstruct(
		World& host_world,
		cudaStream_t& mirror_stream)
	{
		if (host_world.GetCameras().RequiresUpdate() || true)
		{
			cameras.Reconstruct(host_world.GetCameras(), m_hpm, mirror_stream);
			host_world.GetCameras().Updated();
		}
		//pointLights.Reconstruct(host_world.GetPointLights(), m_host_pinned_memory, mirror_stream);
	}
}