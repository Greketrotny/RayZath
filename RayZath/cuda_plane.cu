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

			transformation.position = hPlane->GetPosition();
			transformation.rotation = hPlane->GetRotation();
			transformation.center = hPlane->GetCenter();
			transformation.scale = hPlane->GetScale();
			this->transformation.g2l.ApplyRotationB(-hPlane->GetRotation());
			this->transformation.l2g.ApplyRotation(hPlane->GetRotation());

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