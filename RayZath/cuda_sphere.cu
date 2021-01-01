#include "cuda_sphere.cuh"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		__host__ CudaSphere::CudaSphere()
			: radius(1.0f)
			, material(nullptr)
		{}

		__host__ void CudaSphere::Reconstruct(
			const CudaWorld& hCudaWorld, 
			const Handle<Sphere>& hSphere, 
			cudaStream_t& mirror_stream)
		{
			if (!hSphere->GetStateRegister().IsModified()) return;

			// transposition
			this->position = hSphere->GetPosition();
			this->rotation = hSphere->GetRotation();
			this->center = hSphere->GetCenter();
			this->scale = hSphere->GetScale();
			this->radius = hSphere->GetRadius();

			// bounding box
			this->bounding_box = hSphere->GetBoundingBox();

			// material
			auto& hMaterial = hSphere->GetMaterial();
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


			hSphere->GetStateRegister().MakeUnmodified();
		}
	}
}