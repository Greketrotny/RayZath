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
			reflectance = hMaterial.GetReflectance();
			glossiness = hMaterial.GetReflectance();
			transmittance = hMaterial.GetTransmittance();
			ior = hMaterial.GetIndexOfRefraction();
			emittance = hMaterial.GetEmittance();
			scattering = hMaterial.GetScattering();

			texture = nullptr;
			emittance_map = nullptr;

			return *this;
		}
		void CudaMaterial::Reconstruct(
			const CudaWorld& hCudaWorld,
			const Handle<Material>& hMaterial,
			cudaStream_t& mirror_stream)
		{
			if (!hMaterial->GetStateRegister().IsModified()) return;

			// material properties
			color = hMaterial->GetColor();
			reflectance = hMaterial->GetReflectance();
			glossiness = hMaterial->GetGlossiness();
			transmittance = hMaterial->GetTransmittance();
			ior = hMaterial->GetIndexOfRefraction();
			emittance = hMaterial->GetEmittance();
			scattering = hMaterial->GetScattering();

			// texture
			auto& hTexture = hMaterial->GetTexture();
			if (hTexture)
			{
				if (hTexture.GetResource()->GetId() < hCudaWorld.textures.GetCount())
				{
					texture = hCudaWorld.textures.GetStorageAddress() +
						hTexture.GetResource()->GetId();
				}
				else texture = nullptr;
			}
			else texture = nullptr;

			// emittance map
			auto& hEmittanceMap = hMaterial->GetEmittanceMap();
			if (hEmittanceMap)
			{
				if (hEmittanceMap.GetResource()->GetId() < hCudaWorld.emittance_maps.GetCount())
				{
					emittance_map = hCudaWorld.emittance_maps.GetStorageAddress() +
						hEmittanceMap.GetResource()->GetId();
				}
				else emittance_map = nullptr;
			}
			else emittance_map = nullptr;

			hMaterial->GetStateRegister().MakeUnmodified();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	}
}