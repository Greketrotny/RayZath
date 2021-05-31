#include "cuda_world.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		HostPinnedMemory CudaWorld::m_hpm(0x10000);

		CudaWorld::CudaWorld() 
		{
			// default material
			CudaMaterial* hCudaMaterial = (CudaMaterial*)m_hpm.GetPointerToMemory();
			new (hCudaMaterial) CudaMaterial(
				Color<float>(1.0f, 1.0f, 1.0f, 1.0f),
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);

			CudaErrorCheck(cudaMalloc(&default_material, sizeof(*default_material)));
			CudaErrorCheck(cudaMemcpy(
				default_material, hCudaMaterial,
				sizeof(*default_material),
				cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
		CudaWorld::~CudaWorld()
		{
			// default material
			if (default_material) CudaErrorCheck(cudaFree(default_material));
			default_material = nullptr;
		}

		__host__ void CudaWorld::ReconstructResources(
			World& hWorld,
			cudaStream_t& update_stream)
		{
			ReconstructMaterial(*this, hWorld.GetMaterial(), update_stream);
			ReconstructDefaultMaterial(*this, hWorld.GetDefaultMaterial(), update_stream);

			textures.Reconstruct(*this, hWorld.Container<World::ContainerType::Texture>(), m_hpm, update_stream);
			normal_maps.Reconstruct(*this, hWorld.Container<World::ContainerType::NormalMap>(), m_hpm, update_stream);
			emittance_maps.Reconstruct(*this, hWorld.Container<World::ContainerType::EmittanceMap>(), m_hpm, update_stream);
			reflectance_maps.Reconstruct(*this, hWorld.Container<World::ContainerType::ReflectanceMap>(), m_hpm, update_stream);

			materials.Reconstruct(*this, hWorld.Container<World::ContainerType::Material>(), m_hpm, update_stream);
			mesh_structures.Reconstruct(*this, hWorld.Container<World::ContainerType::MeshStructure>(), m_hpm, update_stream);
		}
		__host__ void CudaWorld::ReconstructObjects(
			World& hWorld,
			cudaStream_t& update_stream)
		{
			pointLights.Reconstruct(*this, hWorld.Container<World::ContainerType::PointLight>(), m_hpm, update_stream);
			spotLights.Reconstruct(*this, hWorld.Container<World::ContainerType::SpotLight>(), m_hpm, update_stream);
			directLights.Reconstruct(*this, hWorld.Container<World::ContainerType::DirectLight>(), m_hpm, update_stream);

			meshes.Reconstruct(*this, hWorld.Container<World::ContainerType::Mesh>(), m_hpm, update_stream);
			spheres.Reconstruct(*this, hWorld.Container<World::ContainerType::Sphere>(), m_hpm, update_stream);
			planes.Reconstruct(*this, hWorld.Container<World::ContainerType::Plane>(), m_hpm, update_stream);
		}
		__host__ void CudaWorld::ReconstructCameras(
			World& hWorld,
			cudaStream_t& update_stream)
		{
			cameras.Reconstruct(*this, hWorld.Container<World::ContainerType::Camera>(), m_hpm, update_stream);
		}
		void CudaWorld::ReconstructAll(
			World& hWorld,
			cudaStream_t& update_stream)
		{
			if (!hWorld.GetStateRegister().IsModified()) return;

			ReconstructResources(hWorld, update_stream);
			ReconstructObjects(hWorld, update_stream);
			ReconstructCameras(hWorld, update_stream);

			hWorld.GetStateRegister().MakeUnmodified();
		}
		__host__ void CudaWorld::ReconstructMaterial(
			const CudaWorld& hCudaWorld,
			const Material& hMaterial,
			cudaStream_t& mirror_stream)
		{
			material = hMaterial;

			// texture
			auto& hTexture = hMaterial.GetTexture();
			if (hTexture)
			{
				if (hTexture.GetResource()->GetId() < hCudaWorld.textures.GetCount())
				{
					material.SetTexture(hCudaWorld.textures.GetStorageAddress() +
						hTexture.GetResource()->GetId());
				}
				else material.SetTexture(nullptr);
			}
			else material.SetTexture(nullptr);
		}
		void CudaWorld::ReconstructDefaultMaterial(
			const CudaWorld& hCudaWorld,
			const Material& hMaterial,
			cudaStream_t& mirror_stream)
		{
			RZAssert(bool(default_material), "default material was nullptr");

			CudaMaterial* hCudaMaterial = (CudaMaterial*)m_hpm.GetPointerToMemory();
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
}