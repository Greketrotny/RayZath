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

			materials.Reconstruct(*this, hWorld.GetMaterials(), m_hpm, mirror_stream);

			cameras.Reconstruct(*this, hWorld.GetCameras(), m_hpm, mirror_stream);

			pointLights.Reconstruct(*this, hWorld.GetPointLights(), m_hpm, mirror_stream);
			spotLights.Reconstruct(*this, hWorld.GetSpotLights(), m_hpm, mirror_stream);
			directLights.Reconstruct(*this, hWorld.GetDirectLights(), m_hpm, mirror_stream);

			meshes.Reconstruct(*this, hWorld.GetMeshes(), m_hpm, mirror_stream);
			spheres.Reconstruct(*this, hWorld.GetSpheres(), m_hpm, mirror_stream);

			material = hWorld.GetMaterial();

			hWorld.GetStateRegister().MakeUnmodified();
		}
	}
}