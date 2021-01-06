#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaMaterial : public WithExistFlag
		{
		private:
		public:
			CudaColor<float> color;

		public:
			float reflectance;
			float glossiness;

			float transmittance;
			float ior;

			float emittance;
			float scattering;

		private:
		public:
			const CudaTexture* texture;

		public:
			__host__ __device__ CudaMaterial(
				const CudaColor<float>& color = CudaColor<float>(1.0f, 1.0f, 1.0f, 1.0f),
				const float& reflectance = 0.0f,
				const float& glossiness = 0.0f,
				const float& transmitance = 1.0f,
				const float& ior = 1.0f,
				const float& emittance = 0.0f,
				const float& scattering = 0.0f)
				: color(color)
				, reflectance(reflectance)
				, glossiness(glossiness)
				, transmittance(transmitance)
				, ior(ior)
				, emittance(emittance)
				, scattering(scattering)
				, texture(nullptr)
			{}

			__host__ CudaMaterial& operator=(const Material& hMaterial);

			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Material>& hMaterial,
				cudaStream_t& mirror_stream);


		public:
			__device__ CudaColor<float> GetColor() const
			{
				return color;
			}
			__device__ CudaColor<float> GetColor(const CudaTexcrd& texcrd) const
			{
				if (texture) return texture->Fetch(texcrd);
				else return color;
			}
		};
	}
}

#endif // !CUDA_MATERIAL_H