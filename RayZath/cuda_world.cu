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
		if (host_world.GetCameras().RequiresUpdate())
			cameras.Reconstruct(host_world.GetCameras(), m_hpm, mirror_stream);

		if (host_world.GetPointLights().RequiresUpdate())
			pointLights.Reconstruct(host_world.GetPointLights(), m_hpm, mirror_stream);
		if (host_world.GetSpotLights().RequiresUpdate())
			spotLights.Reconstruct(host_world.GetSpotLights(), m_hpm, mirror_stream);
		if (host_world.GetDirectLights().RequiresUpdate())
			directLights.Reconstruct(host_world.GetDirectLights(), m_hpm, mirror_stream);
		
		if (host_world.GetMeshes().RequiresUpdate())
			meshes.Reconstruct(host_world.GetMeshes(), m_hpm, mirror_stream);
		if (host_world.GetSpheres().RequiresUpdate())
			spheres.Reconstruct(host_world.GetSpheres(), m_hpm, mirror_stream);
	}
}