#include "cuda_plane.cuh"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		__host__ CudaPlane::CudaPlane()
			: material(nullptr)
		{}

		__host__ void CudaPlane::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Plane>& hPlane,
			cudaStream_t& mirror_stream)
		{
			if (!hPlane->GetStateRegister().IsModified()) return;

			position = hPlane->GetPosition();
			rotation = hPlane->GetRotation();
			center = hPlane->GetCenter();
			scale = hPlane->GetScale();

			// material
			auto& hMaterial = hPlane->GetMaterial();
			if (hMaterial)
			{
				if (hMaterial.GetResource()->GetId() < hCudaWorld.materials.GetCount())
				{
					this->material =
						hCudaWorld.materials.GetStorageAddress() +
						hMaterial.GetResource()->GetId();
				}
				else material = hCudaWorld.default_material;
			}
			else material = hCudaWorld.default_material;

			hPlane->GetStateRegister().MakeUnmodified();
		}
	}
}