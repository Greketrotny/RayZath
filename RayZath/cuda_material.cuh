#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "cuda_buffer.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaMaterial : public WithExistFlag
		{
		private:
			ColorF color;

			float metalic;
			float specular;
			float roughness;
			float emission;

			float ior;
			float scattering;

			const CudaTexture* texture;
			const CudaNormalMap* normal_map;
			const CudaMetalicMap* metalic_map;
			const CudaSpecularMap* specular_map;
			const CudaRoughnessMap* roughness_map;
			const CudaEmissionMap* emission_map;

		public:
			__host__ __device__ CudaMaterial(
				const ColorF& color = ColorF(1.0f),
				const float& metalic = 0.0f,
				const float& specular = 0.0f,
				const float& roughness = 0.0f,
				const float& emission = 0.0f,
				const float& ior = 1.0f,
				const float& scattering = 0.0f)
				: color(color)
				, metalic(metalic)
				, specular(specular)
				, roughness(roughness)
				, emission(emission)
				, ior(ior)
				, scattering(scattering)
				, texture(nullptr)
				, normal_map(nullptr)
				, metalic_map(nullptr)
				, specular_map(nullptr)
				, roughness_map(nullptr)
				, emission_map(nullptr)
			{}

			__host__ CudaMaterial& operator=(const Material& hMaterial);

			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Material>& hMaterial,
				cudaStream_t& mirror_stream);


		public:
			__host__ void SetColor(const Graphics::Color& color)
			{
				this->color = color;
			}
			__host__ void SetTexture(const CudaTexture* texture)
			{
				this->texture = texture;
			}
			__host__ void SetEmission(const float& emission)
			{
				this->emission = emission;
			}

			__device__ const ColorF& GetColor() const
			{
				return color;
			}
			__device__ ColorF GetColor(const CudaTexcrd& texcrd) const
			{
				if (texture) return texture->Fetch(texcrd);
				else return GetColor();
			}
			__device__ const ColorF GetOpacityColor() const
			{
				ColorF color = GetColor();
				color.alpha = 1.0f - color.alpha;
				return color;
			}
			__device__ const ColorF GetOpacityColor(const CudaTexcrd& texcrd) const
			{
				ColorF color = GetColor(texcrd);
				color.alpha = 1.0f - color.alpha;
				return color;
			}
			__device__ const float& GetMetalic() const
			{
				return metalic;
			}
			__device__ float GetMetalic(const CudaTexcrd& texcrd) const
			{
				if (metalic_map) return metalic_map->Fetch(texcrd);
				else return GetMetalic();
			}
			__device__ const float& GetSpecular() const
			{
				return specular;
			}
			__device__ float GetSpecular(const CudaTexcrd& texcrd) const
			{
				if (specular_map) return specular_map->Fetch(texcrd);
				else return GetSpecular();
			}
			__device__ const float& GetRoughness() const
			{
				return roughness;
			}
			__device__ float GetRoughness(const CudaTexcrd& texcrd) const
			{
				if (roughness_map) return roughness_map->Fetch(texcrd);
				else return GetRoughness();
			}
			__device__ const float& GetEmission() const
			{
				return emission;
			}
			__device__ float GetEmission(const CudaTexcrd& texcrd) const
			{
				if (emission_map) return emission_map->Fetch(texcrd);
				else return GetEmission();
			}

			__device__ const float& GetIOR() const
			{
				return ior;
			}
			__device__ const float& GetScattering() const
			{
				return scattering;
			}

			__device__ const CudaNormalMap* GetNormalMap() const
			{
				return normal_map;
			}


		public:
			__device__ bool ApplyScattering(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				if (GetScattering() > 1.0e-4f)
				{
					intersection.ray.length =
						(-cui_logf(rng.GetUnsignedUniform(thread) + 1.0e-4f)) / GetScattering();
					intersection.surface_material = this;
					return true;
				}
				return false;
			}
			__device__ bool SampleDirect(
				const RayIntersection& intersection,
				FullThread& thread,
				const RNG& rng) const
			{
				if (scattering > 0.0f) return true;
				if (intersection.fetched_color.alpha > 0.0f) return false;

				return true;
			}
			__device__ float BRDF(
				const RayIntersection& intersection,
				const vec3f& vPL) const
			{
				if (scattering > 0.0f) return 1.0f;
				if (intersection.fetched_color.alpha > 0.0f) return 0.0f;

				const float vPL_dot_vN = vec3f::DotProduct(
					vPL, intersection.mapped_normal);
				if (vPL_dot_vN <= 0.0f)
					return 0.0f;

				// diffuse reflection
				const float diffuse_brdf = (1.0f - intersection.fetched_specular);

				// specular reflection
				const vec3f vH = HalfwayVector(
					intersection.ray.direction, vPL);
				const float vN_dot_vH = vec3f::DotProduct(intersection.mapped_normal, vH);
				const float specular_brdf =
					cui_powf(
						vN_dot_vH,
						1.0f / (intersection.fetched_roughness + 1.0e-7f)) * 
					intersection.fetched_specular;

				return (diffuse_brdf + specular_brdf) * vPL_dot_vN;
			}


			__device__ float GenerateNextRay(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				if (intersection.fetched_color.alpha > 0.0f)
				{	// ray fell into material/object
					if (intersection.surface_material->scattering > 0.0f)
					{
						return GenerateScatteringRay(thread, intersection, rng);
					}
					else
					{
						return GenerateTransmissiveRay(thread, intersection, rng);
					}
				}
				else
				{	// ray is reflected from sufrace

					if (rng.GetUnsignedUniform(thread) > intersection.fetched_specular)
					{	// diffuse reflection
						return GenerateDiffuseRay(thread, intersection, rng);
					}
					else
					{	// glossy reflection
						return GenerateGlossyRay(thread, intersection, rng);
					}
				}
			}


		private:
			__device__ float GenerateDiffuseRay(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				vec3f sample = CosineSampleHemisphere(
					rng.GetUnsignedUniform(thread),
					rng.GetUnsignedUniform(thread),
					intersection.mapped_normal);
				sample.Normalize();

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(sample, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) sample += intersection.surface_normal * -2.0f * vR_dot_vN;

				new (&intersection.ray) CudaSceneRay(
					intersection.point + intersection.surface_normal * 0.0001f,
					sample,
					intersection.ray.material);

				return 1.0f;
			}
			__device__ float GenerateGlossyRay(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				const vec3f vH = SampleHemisphere(
					rng.GetUnsignedUniform(thread),
					1.0f - cui_powf(
						rng.GetUnsignedUniform(thread),
						intersection.fetched_roughness + 1.0e-5f),
					intersection.mapped_normal);

				// calculate reflection direction
				vec3f vR = ReflectVector(
					intersection.ray.direction,
					vH);

				// reflect sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(vR, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

				// create next glossy CudaSceneRay
				new (&intersection.ray) CudaSceneRay(
					intersection.point + intersection.surface_normal * 0.0001f,
					vR,
					intersection.ray.material);

				return intersection.fetched_metalic;
			}
			__device__ float GenerateTransmissiveRay(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				if (intersection.behind_material->ior != intersection.ray.material->ior)
				{	// refraction ray

					const float cosi = fabsf(vec3f::DotProduct(
						intersection.ray.direction, intersection.mapped_normal));

					// calculate sin^2 theta from Snell's law
					const float& n1 = intersection.ray.material->ior;
					const float& n2 = intersection.behind_material->ior;
					const float ratio = n1 / n2;
					const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);

					if (sin2_t >= 1.0f)
					{	// TIR

						// calculate reflection vector
						vec3f vR = ReflectVector(
							intersection.ray.direction,
							intersection.mapped_normal);

						// flip sample above surface if needed
						const float vR_dot_vN = vec3f::DotProduct(vR, intersection.surface_normal);
						if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

						// create new internal reflection CudaSceneRay
						new (&intersection.ray) CudaSceneRay(
							intersection.point + intersection.surface_normal * 0.0001f,
							vR,
							intersection.ray.material);

						return 1.0f;
					}
					else
					{
						// calculate fresnel
						const float cost = sqrtf(1.0f - sin2_t);
						const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
						const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
						const float f = (Rs * Rs + Rp * Rp) / 2.0f;

						if (f < rng.GetUnsignedUniform(thread))
						{	// transmission/refraction

							// calculate refraction direction
							const vec3f vR = intersection.ray.direction * ratio +
								intersection.mapped_normal * (ratio * cosi - cost);

							// create new refraction CudaSceneRay
							new (&intersection.ray) CudaSceneRay(
								intersection.point - intersection.surface_normal * 0.0001f,
								vR,
								intersection.behind_material);

							return 1.0f;
						}
						else
						{	// reflection

							// calculate reflection direction
							vec3f vR = ReflectVector(
								intersection.ray.direction,
								intersection.mapped_normal);

							// flip sample above surface if needed
							const float vR_dot_vN = vec3f::DotProduct(vR, intersection.surface_normal);
							if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

							// create new reflection CudaSceneRay
							new (&intersection.ray) CudaSceneRay(
								intersection.point + intersection.surface_normal * 0.0001f,
								vR,
								intersection.ray.material);

							return intersection.fetched_metalic;
						}
					}
				}
				else
				{	// transparent ray

					vec3f vD;

					if (intersection.behind_material->roughness > 0.0f)
					{
						vD = SampleSphere(
							rng.GetUnsignedUniform(thread),
							1.0f - cui_powf(
								rng.GetUnsignedUniform(thread),
								intersection.behind_material->roughness),
							intersection.ray.direction);

						const float vS_dot_vN = vec3f::DotProduct(vD, -intersection.surface_normal);
						if (vS_dot_vN < 0.0f) vD += -intersection.surface_normal * -2.0f * vS_dot_vN;
					}
					else
					{
						vD = intersection.ray.direction;
					}

					new (&intersection.ray) CudaSceneRay(
						intersection.point - intersection.surface_normal * 0.0001f,
						vD,
						intersection.behind_material);

					return 1.0f;
				}
			}
			__device__ float GenerateScatteringRay(
				FullThread& thread,
				RayIntersection& intersection,
				const RNG& rng) const
			{
				// generate scatter direction
				const vec3f sctr_direction = SampleSphere(
					rng.GetUnsignedUniform(thread),
					rng.GetUnsignedUniform(thread),
					intersection.ray.direction);

				// create scattering ray
				new (&intersection.ray) CudaSceneRay(
					intersection.point,
					sctr_direction,
					intersection.ray.material);

				return intersection.fetched_metalic;
			}
		};
	}
}

#endif // !CUDA_MATERIAL_H