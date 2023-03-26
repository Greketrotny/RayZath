#include "cuda_material.cuh"
#include "cuda_world.cuh"

#include "material.hpp"

namespace RayZath::Cuda
{
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
		if (hMaterial->map<Engine::ObjectType::Texture>())
		{
			if (hMaterial->map<Engine::ObjectType::Texture>().accessor()->idx() < hCudaWorld.textures.count())
			{
				mp_texture = hCudaWorld.textures.storageAddress() +
					hMaterial->map<Engine::ObjectType::Texture>().accessor()->idx();
			}
		}

		// normal map
		if (hMaterial->map<Engine::ObjectType::NormalMap>())
		{
			if (hMaterial->map<Engine::ObjectType::NormalMap>().accessor()->idx() < hCudaWorld.normal_maps.count())
			{
				mp_normal_map = hCudaWorld.normal_maps.storageAddress() +
					hMaterial->map<Engine::ObjectType::NormalMap>().accessor()->idx();
			}
		}

		// metalness map
		if (hMaterial->map<Engine::ObjectType::MetalnessMap>())
		{
			if (hMaterial->map<Engine::ObjectType::MetalnessMap>().accessor()->idx() < hCudaWorld.metalness_maps.count())
			{
				mp_metalness_map = hCudaWorld.metalness_maps.storageAddress() +
					hMaterial->map<Engine::ObjectType::MetalnessMap>().accessor()->idx();
			}
		}

		// roughness map
		if (hMaterial->map<Engine::ObjectType::RoughnessMap>())
		{
			if (hMaterial->map<Engine::ObjectType::RoughnessMap>().accessor()->idx() < hCudaWorld.roughness_maps.count())
			{
				mp_roughness_map = hCudaWorld.roughness_maps.storageAddress() +
					hMaterial->map<Engine::ObjectType::RoughnessMap>().accessor()->idx();
			}
		}

		// emission map
		if (hMaterial->map<Engine::ObjectType::EmissionMap>())
		{
			if (hMaterial->map<Engine::ObjectType::EmissionMap>().accessor()->idx() < hCudaWorld.emission_maps.count())
			{
				mp_emission_map = hCudaWorld.emission_maps.storageAddress() +
					hMaterial->map<Engine::ObjectType::EmissionMap>().accessor()->idx();
			}
		}

		hMaterial->stateRegister().MakeUnmodified();
	}
}