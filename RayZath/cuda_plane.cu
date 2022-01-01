#include "cuda_plane.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	__host__ Plane::Plane()
		: material(nullptr)
	{}

	__host__ void Plane::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::Plane>& hPlane,
		cudaStream_t& mirror_stream)
	{
		if (!hPlane->GetStateRegister().IsModified()) return;

		// transformation
		transformation = hPlane->GetTransformation();

		// material
		auto& hMaterial = hPlane->GetMaterial();
		if (hMaterial)
		{
			if (hMaterial.GetAccessor()->GetIdx() < hCudaWorld.materials.GetCount())
			{
				this->material =
					hCudaWorld.materials.GetStorageAddress() +
					hMaterial.GetAccessor()->GetIdx();
			}
			else material = hCudaWorld.default_material;
		}
		else material = hCudaWorld.default_material;

		hPlane->GetStateRegister().MakeUnmodified();
	}
}