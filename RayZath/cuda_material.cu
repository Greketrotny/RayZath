#include "cuda_material.cuh"
#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		// ~~~~~~~~ [STRUCT] CudaMaterial ~~~~~~~~
		CudaMaterial& CudaMaterial::operator=(const Material& hMaterial)
		{
			m_color = hMaterial.GetColor();
			m_metalness = hMaterial.GetMetalness();
			m_roughness = hMaterial.GetRoughness();
			m_emission = hMaterial.GetEmission();
			m_ior = hMaterial.GetIOR();
			m_scattering = hMaterial.GetScattering();

			mp_texture = nullptr;
			mp_normal_map = nullptr;
			mp_metalness_map = nullptr;
			mp_roughness_map = nullptr;
			mp_emission_map = nullptr;

			return *this;
		}
		void CudaMaterial::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Material>& hMaterial,
			cudaStream_t& mirror_stream)
		{
			if (!hMaterial->GetStateRegister().IsModified()) return;

			// material properties
			*this = *hMaterial.GetAccessor()->Get();

			// texture
			if (hMaterial->GetTexture())
			{
				if (hMaterial->GetTexture().GetAccessor()->GetIdx() < hCudaWorld.textures.GetCount())
				{
					mp_texture = hCudaWorld.textures.GetStorageAddress() +
						hMaterial->GetTexture().GetAccessor()->GetIdx();
				}
			}

			// normal map
			if (hMaterial->GetNormalMap())
			{
				if (hMaterial->GetNormalMap().GetAccessor()->GetIdx() < hCudaWorld.normal_maps.GetCount())
				{
					mp_normal_map = hCudaWorld.normal_maps.GetStorageAddress() +
						hMaterial->GetNormalMap().GetAccessor()->GetIdx();
				}
			}

			// metalness map
			if (hMaterial->GetMetalnessMap())
			{
				if (hMaterial->GetMetalnessMap().GetAccessor()->GetIdx() < hCudaWorld.metalness_maps.GetCount())
				{
					mp_metalness_map = hCudaWorld.metalness_maps.GetStorageAddress() +
						hMaterial->GetMetalnessMap().GetAccessor()->GetIdx();
				}
			}

			// roughness map
			if (hMaterial->GetRoughnessMap())
			{
				if (hMaterial->GetRoughnessMap().GetAccessor()->GetIdx() < hCudaWorld.roughness_maps.GetCount())
				{
					mp_roughness_map = hCudaWorld.roughness_maps.GetStorageAddress() +
						hMaterial->GetRoughnessMap().GetAccessor()->GetIdx();
				}
			}

			// emission map
			if (hMaterial->GetEmissionMap())
			{
				if (hMaterial->GetEmissionMap().GetAccessor()->GetIdx() < hCudaWorld.emission_maps.GetCount())
				{
					mp_emission_map = hCudaWorld.emission_maps.GetStorageAddress() +
						hMaterial->GetEmissionMap().GetAccessor()->GetIdx();
				}
			}

			hMaterial->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}