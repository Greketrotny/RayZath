#include "cuda_sphere.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	__host__ Sphere::Sphere()
		: radius(1.0f)
		, material(nullptr)
	{}

	__host__ void Sphere::Reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::Sphere>& hSphere,
		cudaStream_t& mirror_stream)
	{
		if (!hSphere->GetStateRegister().IsModified()) return;

		radius = hSphere->GetRadius();
		transformation = hSphere->GetTransformation();
		bounding_box = hSphere->GetBoundingBox();

		// material
		auto& hMaterial = hSphere->GetMaterial();
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


		hSphere->GetStateRegister().MakeUnmodified();
	}
}