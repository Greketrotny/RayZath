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
				CudaColor<float>(1.0f, 1.0f, 1.0f, 1.0f),
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

		void CudaWorld::Reconstruct(
			World& hWorld,
			cudaStream_t& mirror_stream)
		{
			if (!hWorld.GetStateRegister().IsModified()) return;

			textures.Reconstruct(*this, hWorld.Container<Texture>(), m_hpm, mirror_stream);
			materials.Reconstruct(*this, hWorld.Container<Material>(), m_hpm, mirror_stream);
			mesh_structures.Reconstruct(*this, hWorld.Container<MeshStructure>(), m_hpm, mirror_stream);

			cameras.Reconstruct(*this, hWorld.Container<Camera>(), m_hpm, mirror_stream);

			pointLights.Reconstruct(*this, hWorld.Container<PointLight>(), m_hpm, mirror_stream);
			spotLights.Reconstruct(*this, hWorld.Container<SpotLight>(), m_hpm, mirror_stream);
			directLights.Reconstruct(*this, hWorld.Container<DirectLight>(), m_hpm, mirror_stream);

			meshes.Reconstruct(*this, hWorld.Container<Mesh>(), m_hpm, mirror_stream);
			spheres.Reconstruct(*this, hWorld.Container<Sphere>(), m_hpm, mirror_stream);
			planes.Reconstruct(*this, hWorld.Container<Plane>(), m_hpm, mirror_stream);

			ReconstructMaterial(*this, hWorld.GetMaterial(), mirror_stream);
			ReconstructDefaultMaterial(*this, hWorld.GetDefaultMaterial(), mirror_stream);

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
			RZAssert(bool(default_material), L"default material was nullptr");

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