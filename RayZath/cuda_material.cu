#include "cuda_material.cuh"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [STRUCT] CudaMaterial ~~~~~~~~
		CudaMaterial& CudaMaterial::operator=(const Material& hMaterial)
		{
			color = hMaterial.GetColor();
			metalness = hMaterial.GetMetalness();
			specularity = hMaterial.GetSpecularity();
			roughness = hMaterial.GetRoughness();
			emission = hMaterial.GetEmission();
			ior = hMaterial.GetIOR();
			scattering = hMaterial.GetScattering();

			texture = nullptr;
			normal_map = nullptr;
			metalness_map = nullptr;
			specularity_map = nullptr;
			roughness_map = nullptr;
			emission_map = nullptr;

			return *this;
		}
		void CudaMaterial::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Material>& hMaterial,
			cudaStream_t& mirror_stream)
		{
			if (!hMaterial->GetStateRegister().IsModified()) return;

			// material properties
			*this = *hMaterial.GetResource()->GetData();

			// texture
			if (hMaterial->GetTexture())
			{
				if (hMaterial->GetTexture().GetResource()->GetId() < hCudaWorld.textures.GetCount())
				{
					texture = hCudaWorld.textures.GetStorageAddress() +
						hMaterial->GetTexture().GetResource()->GetId();
				}
			}

			// normal map
			if (hMaterial->GetNormalMap())
			{
				if (hMaterial->GetNormalMap().GetResource()->GetId() < hCudaWorld.normal_maps.GetCount())
				{
					normal_map = hCudaWorld.normal_maps.GetStorageAddress() +
						hMaterial->GetNormalMap().GetResource()->GetId();
				}
			}

			// metalness map
			if (hMaterial->GetMetalnessMap())
			{
				if (hMaterial->GetMetalnessMap().GetResource()->GetId() < hCudaWorld.metalness_maps.GetCount())
				{
					metalness_map = hCudaWorld.metalness_maps.GetStorageAddress() +
						hMaterial->GetMetalnessMap().GetResource()->GetId();
				}
			}

			// specularity map
			if (hMaterial->GetSpecularityMap())
			{
				if (hMaterial->GetSpecularityMap().GetResource()->GetId() < hCudaWorld.specularity_maps.GetCount())
				{
					specularity_map = hCudaWorld.specularity_maps.GetStorageAddress() +
						hMaterial->GetSpecularityMap().GetResource()->GetId();
				}
			}

			// roughness map
			if (hMaterial->GetRoughnessMap())
			{
				if (hMaterial->GetRoughnessMap().GetResource()->GetId() < hCudaWorld.roughness_maps.GetCount())
				{
					roughness_map = hCudaWorld.roughness_maps.GetStorageAddress() +
						hMaterial->GetRoughnessMap().GetResource()->GetId();
				}
			}

			// emittance map
			if (hMaterial->GetEmissionMap())
			{
				if (hMaterial->GetEmissionMap().GetResource()->GetId() < hCudaWorld.emission_maps.GetCount())
				{
					emission_map = hCudaWorld.emission_maps.GetStorageAddress() +
						hMaterial->GetEmissionMap().GetResource()->GetId();
				}
			}

			hMaterial->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}