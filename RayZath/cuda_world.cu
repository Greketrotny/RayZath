#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaWorld::m_hpm(0xFFFF);

		CudaWorld::CudaWorld()
		{}
		CudaWorld::~CudaWorld()
		{}

		void CudaWorld::Reconstruct(
			World& hWorld,
			cudaStream_t& mirror_stream)
		{
			if (!hWorld.GetStateRegister().IsModified()) return;

			materials.Reconstruct(hWorld.GetMaterials(), m_hpm, mirror_stream);

			cameras.Reconstruct(hWorld.GetCameras(), m_hpm, mirror_stream);

			pointLights.Reconstruct(hWorld.GetPointLights(), m_hpm, mirror_stream);
			spotLights.Reconstruct(hWorld.GetSpotLights(), m_hpm, mirror_stream);
			directLights.Reconstruct(hWorld.GetDirectLights(), m_hpm, mirror_stream);

			meshes.Reconstruct(hWorld.GetMeshes(), m_hpm, mirror_stream);
			spheres.Reconstruct(hWorld.GetSpheres(), m_hpm, mirror_stream);

			material = hWorld.GetMaterial();

			hWorld.GetStateRegister().MakeUnmodified();
		}
	}
}