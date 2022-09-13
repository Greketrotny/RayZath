#include "cuda_material.cuh"
#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	// ~~~~~~~~ [STRUCT] Material ~~~~~~~~
	Material& Material::operator=(const RayZath::Engine::Material& hMaterial)
	{
		m_color = hMaterial.color();
		m_metalness = hMaterial.metalness();
		m_roughness = hMaterial.roughness();
		m_emission = hMaterial.emission();
		m_ior = hMaterial.ior();
		m_scattering = hMaterial.scattering();

		mp_texture = nullptr;
		mp_normal_map = nullptr;
		mp_metalness_map = nullptr;
		mp_roughness_map = nullptr;
		mp_emission_map = nullptr;

		return *this;
	}
	void Material::reconstruct(
		const World& hCudaWorld,
		const RayZath::Engine::Handle<RayZath::Engine::Material>& hMaterial,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		if (!hMaterial->stateRegister().IsModified()) return;

		// material properties
		*this = *hMaterial.accessor()->get();

		// texture
		if (hMaterial->texture())
		{
			if (hMaterial->texture().accessor()->idx() < hCudaWorld.textures.count())
			{
				mp_texture = hCudaWorld.textures.storageAddress() +
					hMaterial->texture().accessor()->idx();
			}
		}

		// normal map
		if (hMaterial->normalMap())
		{
			if (hMaterial->normalMap().accessor()->idx() < hCudaWorld.normal_maps.count())
			{
				mp_normal_map = hCudaWorld.normal_maps.storageAddress() +
					hMaterial->normalMap().accessor()->idx();
			}
		}

		// metalness map
		if (hMaterial->metalnessMap())
		{
			if (hMaterial->metalnessMap().accessor()->idx() < hCudaWorld.metalness_maps.count())
			{
				mp_metalness_map = hCudaWorld.metalness_maps.storageAddress() +
					hMaterial->metalnessMap().accessor()->idx();
			}
		}

		// roughness map
		if (hMaterial->roughnessMap())
		{
			if (hMaterial->roughnessMap().accessor()->idx() < hCudaWorld.roughness_maps.count())
			{
				mp_roughness_map = hCudaWorld.roughness_maps.storageAddress() +
					hMaterial->roughnessMap().accessor()->idx();
			}
		}

		// emission map
		if (hMaterial->emissionMap())
		{
			if (hMaterial->emissionMap().accessor()->idx() < hCudaWorld.emission_maps.count())
			{
				mp_emission_map = hCudaWorld.emission_maps.storageAddress() +
					hMaterial->emissionMap().accessor()->idx();
			}
		}

		hMaterial->stateRegister().MakeUnmodified();
	}
}