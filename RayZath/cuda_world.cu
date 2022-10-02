#include "cuda_world.cuh"

namespace RayZath::Cuda
{
	HostPinnedMemory World::m_hpm(0x10000);

	World::World()
	{
		// default material
		Material* hCudaMaterial = (Material*)m_hpm.GetPointerToMemory();
		new (hCudaMaterial) Material(
			Color<float>(1.0f, 1.0f, 1.0f, 1.0f),
			0.0f, 0.0f, 1.0f, 0.0f, 0.0f);

		RZAssertCoreCUDA(cudaMalloc(&default_material, sizeof(*default_material)));
		RZAssertCoreCUDA(cudaMemcpy(
			default_material, hCudaMaterial,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
	World::~World()
	{
		// default material
		if (default_material) RZAssertCoreCUDA(cudaFree(default_material));
		default_material = nullptr;
	}

	__host__ void World::reconstructResources(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		reconstructMaterial(hWorld.material(), update_stream);
		reconstructDefaultMaterial(hWorld.defaultMaterial(), update_stream);

		textures.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::Texture>(), m_hpm, update_stream);
		normal_maps.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::NormalMap>(), m_hpm, update_stream);
		metalness_maps.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::MetalnessMap>(), m_hpm, update_stream);
		roughness_maps.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::RoughnessMap>(), m_hpm, update_stream);
		emission_maps.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::EmissionMap>(), m_hpm, update_stream);

		materials.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::Material>(), m_hpm, update_stream);
		mesh_structures.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::MeshStructure>(), m_hpm, update_stream);
	}
	__host__ void World::reconstructObjects(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		cudaStream_t& update_stream)
	{
		spot_lights.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::SpotLight>(), m_hpm, update_stream);
		direct_lights.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::DirectLight>(), m_hpm, update_stream);

		meshes.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::Mesh>(), m_hpm, update_stream);

		sample_direct_light = !direct_lights.empty() && render_config.lightSampling().directLight() != 0u;
		sample_spot_light = !spot_lights.empty() && render_config.lightSampling().spotLight() != 0u;
		sample_direct = sample_direct_light || sample_spot_light;
	}
	__host__ void World::reconstructCameras(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		cameras.reconstruct(*this, hWorld.container<RayZath::Engine::World::ObjectType::Camera>(), m_hpm, update_stream);
	}
	void World::reconstructAll(
		RayZath::Engine::World& hWorld,
		const RayZath::Engine::RenderConfig& render_config,
		cudaStream_t& update_stream)
	{
		if (!hWorld.stateRegister().IsModified()) return;

		reconstructResources(hWorld, update_stream);
		reconstructObjects(hWorld, render_config, update_stream);
		reconstructCameras(hWorld, update_stream);

		hWorld.stateRegister().MakeUnmodified();
	}
	__host__ void World::reconstructMaterial(
		const RayZath::Engine::Material& hMaterial,
		[[maybe_unused]] cudaStream_t& mirror_stream)
	{
		material = hMaterial;

		// texture
		if (auto& hTexture = hMaterial.texture();
			hTexture &&
			hTexture.accessor()->idx() < textures.count())
		{
			material.texture(textures.storageAddress() + hTexture.accessor()->idx());
		}
		else material.texture(nullptr);

		// emission map
		if (auto& hEmissionMap = hMaterial.emissionMap();
			hEmissionMap &&
			hEmissionMap.accessor()->idx() < emission_maps.count())
		{
			material.emission_map(emission_maps.storageAddress() + hEmissionMap.accessor()->idx());
		}
		else material.emission_map(nullptr);
	}
	void World::reconstructDefaultMaterial(
		const RayZath::Engine::Material& hMaterial,
		cudaStream_t& mirror_stream)
	{
		RZAssert(bool(default_material), "default material was nullptr");

		Material* hCudaMaterial = (Material*)m_hpm.GetPointerToMemory();
		RZAssertCoreCUDA(cudaMemcpyAsync(
			hCudaMaterial, default_material,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));

		*hCudaMaterial = hMaterial;

		RZAssertCoreCUDA(cudaMemcpyAsync(
			default_material, hCudaMaterial,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
		RZAssertCoreCUDA(cudaStreamSynchronize(mirror_stream));
	}
}