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

		CudaErrorCheck(cudaMalloc(&default_material, sizeof(*default_material)));
		CudaErrorCheck(cudaMemcpy(
			default_material, hCudaMaterial,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
	World::~World()
	{
		// default material
		if (default_material) CudaErrorCheck(cudaFree(default_material));
		default_material = nullptr;
	}

	__host__ void World::ReconstructResources(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		ReconstructMaterial(hWorld.GetMaterial(), update_stream);
		ReconstructDefaultMaterial(hWorld.GetDefaultMaterial(), update_stream);

		textures.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Texture>(), m_hpm, update_stream);
		normal_maps.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::NormalMap>(), m_hpm, update_stream);
		metalness_maps.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::MetalnessMap>(), m_hpm, update_stream);
		roughness_maps.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::RoughnessMap>(), m_hpm, update_stream);
		emission_maps.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::EmissionMap>(), m_hpm, update_stream);

		materials.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Material>(), m_hpm, update_stream);
		mesh_structures.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::MeshStructure>(), m_hpm, update_stream);
	}
	__host__ void World::ReconstructObjects(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		point_lights.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::PointLight>(), m_hpm, update_stream);
		spot_lights.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::SpotLight>(), m_hpm, update_stream);
		direct_lights.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::DirectLight>(), m_hpm, update_stream);

		meshes.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Mesh>(), m_hpm, update_stream);
		spheres.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Sphere>(), m_hpm, update_stream);
		planes.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Plane>(), m_hpm, update_stream);
	}
	__host__ void World::ReconstructCameras(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		cameras.Reconstruct(*this, hWorld.Container<RayZath::Engine::World::ContainerType::Camera>(), m_hpm, update_stream);
	}
	void World::ReconstructAll(
		RayZath::Engine::World& hWorld,
		cudaStream_t& update_stream)
	{
		if (!hWorld.GetStateRegister().IsModified()) return;

		ReconstructResources(hWorld, update_stream);
		ReconstructObjects(hWorld, update_stream);
		ReconstructCameras(hWorld, update_stream);

		hWorld.GetStateRegister().MakeUnmodified();
	}
	__host__ void World::ReconstructMaterial(
		const RayZath::Engine::Material& hMaterial,
		cudaStream_t& mirror_stream)
	{
		material = hMaterial;

		// texture
		auto& hTexture = hMaterial.GetTexture();
		if (hTexture)
		{
			if (hTexture.GetAccessor()->GetIdx() < textures.GetCount())
			{
				material.SetTexture(textures.GetStorageAddress() +
					hTexture.GetAccessor()->GetIdx());
			}
			else material.SetTexture(nullptr);
		}
		else material.SetTexture(nullptr);
	}
	void World::ReconstructDefaultMaterial(
		const RayZath::Engine::Material& hMaterial,
		cudaStream_t& mirror_stream)
	{
		RZAssert(bool(default_material), "default material was nullptr");

		Material* hCudaMaterial = (Material*)m_hpm.GetPointerToMemory();
		CudaErrorCheck(cudaMemcpyAsync(
			hCudaMaterial, default_material,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyDeviceToHost, mirror_stream));
		CudaErrorCheck(cudaStreamSynchronize(mirror_stream));

		*hCudaMaterial = hMaterial;

		CudaErrorCheck(cudaMemcpyAsync(
			default_material, hCudaMaterial,
			sizeof(*default_material),
			cudaMemcpyKind::cudaMemcpyHostToDevice, mirror_stream));
		CudaErrorCheck(cudaStreamSynchronize(mirror_stream));
	}
}