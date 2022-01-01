#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "cuda_buffer.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath
{
	namespace CudaEngine
	{
		struct CudaMaterial
		{
		private:
			ColorF m_color;

			float m_metalness;
			float m_roughness;
			float m_emission;

			float m_ior;
			float m_scattering;

			const CudaTexture* mp_texture;
			const CudaNormalMap* mp_normal_map;
			const CudaMetalnessMap* mp_metalness_map;
			const CudaRoughnessMap* mp_roughness_map;
			const CudaEmissionMap* mp_emission_map;

		public:
			__host__ __device__ CudaMaterial(
				const ColorF& color = ColorF(1.0f),
				const float metalness = 0.0f,
				const float roughness = 0.0f,
				const float emission = 0.0f,
				const float ior = 1.0f,
				const float scattering = 0.0f)
				: m_color(color)
				, m_metalness(metalness)
				, m_roughness(roughness)
				, m_emission(emission)
				, m_ior(ior)
				, m_scattering(scattering)
				, mp_texture(nullptr)
				, mp_normal_map(nullptr)
				, mp_metalness_map(nullptr)
				, mp_roughness_map(nullptr)
				, mp_emission_map(nullptr)
			{}

			__host__ CudaMaterial& operator=(const Material& hMaterial);

			__host__ void Reconstruct(
				const CudaWorld& hCudaWorld,
				const Handle<Material>& hMaterial,
				cudaStream_t& mirror_stream);


		public:
			__host__ void SetColor(const Graphics::Color& color)
			{
				m_color = color;
			}
			__host__ void SetTexture(const CudaTexture* texture)
			{
				mp_texture = texture;
			}
			__host__ void SetEmission(const float emission)
			{
				m_emission = emission;
			}

			__device__ const ColorF& GetColor() const
			{
				return m_color;
			}
			__device__ ColorF GetColor(const CudaTexcrd texcrd) const
			{
				if (mp_texture) return mp_texture->Fetch(texcrd);
				else return GetColor();
			}
			__device__ const ColorF GetOpacityColor() const
			{
				ColorF color = GetColor();
				color.alpha = 1.0f - color.alpha;
				return color;
			}
			__device__ const ColorF GetOpacityColor(const CudaTexcrd texcrd) const
			{
				ColorF color = GetColor(texcrd);
				color.alpha = 1.0f - color.alpha;
				return color;
			}
			__device__ float GetMetalness() const
			{
				return m_metalness;
			}
			__device__ float GetMetalness(const CudaTexcrd texcrd) const
			{
				if (mp_metalness_map) return mp_metalness_map->Fetch(texcrd);
				else return GetMetalness();
			}
			__device__ float GetRoughness() const
			{
				return m_roughness;
			}
			__device__ float GetRoughness(const CudaTexcrd texcrd) const
			{
				if (mp_roughness_map) return mp_roughness_map->Fetch(texcrd);
				else return GetRoughness();
			}
			__device__ float GetEmission() const
			{
				return m_emission;
			}
			__device__ float GetEmission(const CudaTexcrd texcrd) const
			{
				if (mp_emission_map) return mp_emission_map->Fetch(texcrd);
				else return GetEmission();
			}

			__device__ float GetIOR() const
			{
				return m_ior;
			}
			__device__ float GetScattering() const
			{
				return m_scattering;
			}

			__device__ const CudaNormalMap* GetNormalMap() const
			{
				return mp_normal_map;
			}

		private:
			__device__ float NormalDistribution(
				const vec3f vN, const vec3f vH, 
				const float roughness) const
			{
				const float half_angle = vec3f::DotProduct(vN, vH);
				const float b = (half_angle * half_angle) * (roughness - 1.0f) + 1.0001f;
				return (roughness / (b * b));
			}

		public:
			__device__ bool ApplyScattering(
				RayIntersection& intersection,
				RNG& rng) const
			{
				if (GetScattering() > 1.0e-4f)
				{
					const float scatter_distance =
						(-cui_logf(rng.UnsignedUniform() + 1.0e-4f)) / GetScattering();
					if (scatter_distance < intersection.ray.near_far.y)
					{
						intersection.ray.near_far.y = scatter_distance;
						intersection.surface_material = this;
						return true;
					}
				}
				return false;
			}
			__device__ bool SampleDirect(
				const RayIntersection& intersection) const
			{
				return intersection.color.alpha == 0.0f;
			}
			__device__ ColorF BRDF(
				const RayIntersection& intersection,
				const vec3f& vPL) const
			{
				if (intersection.surface_material->GetScattering() > 0.0f) return ColorF(1.0f);

				const float vN_dot_vPL = vec3f::DotProduct(intersection.mapped_normal, vPL);
				const vec3f vH = HalfwayVector(intersection.ray.direction, vPL);

				const float distr = NormalDistribution(
					intersection.mapped_normal,
					vH,
					intersection.roughness);

				return Lerp(intersection.color,
					ColorF(1.0f) * distr,
					intersection.reflectance) * vN_dot_vPL * (1.0f / CUDART_PI);
			}


			// ray generation
			__device__ float GenerateNextRay(
				RayIntersection& intersection,
				RNG& rng) const
			{
				if (intersection.color.alpha > 0.0f)
				{	// transmission
					if (intersection.surface_material->m_scattering > 0.0f)
					{
						return GenerateScatteringRay(intersection, rng);
					}
					else
					{
						return GenerateTransmissiveRay(intersection, rng);
					}
				}
				else
				{	// reflection

					if (rng.UnsignedUniform() > intersection.reflectance)
					{	// diffuse reflection
						return GenerateDiffuseRay(intersection, rng);
					}
					else
					{	// glossy reflection
						return GenerateGlossyRay(intersection, rng);
					}
				}
			}
		private:
			__device__ float GenerateDiffuseRay(
				RayIntersection& intersection,
				RNG& rng) const
			{
				vec3f sample = CosineSampleHemisphere(
					rng.UnsignedUniform(),
					rng.UnsignedUniform(),
					intersection.mapped_normal);
				sample.Normalize();

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(sample, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) sample += intersection.surface_normal * -2.0f * vR_dot_vN;

				intersection.ray.origin =
					intersection.point + intersection.surface_normal * 0.0001f;
				intersection.ray.direction = sample;
				intersection.ray.ResetRange();

				return 1.0f;
			}
			__device__ float GenerateGlossyRay(
				RayIntersection& intersection,
				RNG& rng) const
			{
				const vec3f vH = SampleHemisphere(
					rng.UnsignedUniform(),
					1.0f - cui_powf(
						rng.UnsignedUniform() + 1.0e-5f,
						intersection.roughness),
					intersection.mapped_normal);

				// calculate reflection direction
				vec3f vR = ReflectVector(
					intersection.ray.direction,
					vH);

				// reflect sample above surface if needed
				const float vR_dot_vN = vec3f::Similarity(vR, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) vR += intersection.surface_normal * -2.0f * vR_dot_vN;

				// create next glossy CudaSceneRay
				intersection.ray.origin =
					intersection.point + intersection.surface_normal * 0.0001f;
				intersection.ray.direction = vR;
				intersection.ray.ResetRange();

				return intersection.metalness;
			}
			__device__ float GenerateTransmissiveRay(
				RayIntersection& intersection,
				RNG& rng) const
			{
				const float cosi = fabsf(vec3f::DotProduct(
					intersection.ray.direction, intersection.mapped_normal));

				// calculate sin^2 theta from Snell's law
				const float n1 = intersection.ray.material->m_ior;
				const float n2 = intersection.behind_material->m_ior;
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

					// set reflection ray
					intersection.ray.origin =
						intersection.point + intersection.surface_normal * 0.0001f;
					intersection.ray.direction = vR;
					intersection.ray.ResetRange();

					return 1.0f;
				}
				else
				{
					// calculate fresnel
					const float cost = sqrtf(1.0f - sin2_t);
					const float Rp = ((n1 * cosi) - (n2 * cost)) / ((n1 * cosi) + (n2 * cost));
					const float Rs = ((n2 * cosi) - (n1 * cost)) / ((n2 * cosi) + (n1 * cost));
					const float f = (Rs * Rs + Rp * Rp) / 2.0f;

					if (f < rng.UnsignedUniform())
					{	// transmission/refraction

						// calculate refraction direction
						const vec3f vR = intersection.ray.direction * ratio +
							intersection.mapped_normal * (ratio * cosi - cost);

						// set refraction ray
						intersection.ray.origin =
							intersection.point - intersection.surface_normal * 0.0001f;
						intersection.ray.direction = vR;
						intersection.ray.material = intersection.behind_material;
						intersection.ray.ResetRange();

						return intersection.metalness;
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
						intersection.ray.origin =
							intersection.point + intersection.surface_normal * 0.0001f;
						intersection.ray.direction = vR;
						intersection.ray.ResetRange();

						return intersection.metalness;
					}
				}
			}
			__device__ float GenerateScatteringRay(
				RayIntersection& intersection,
				RNG& rng) const
			{
				// generate scatter direction
				const vec3f sctr_direction = SampleSphere(
					rng.UnsignedUniform(),
					rng.UnsignedUniform(),
					intersection.ray.direction);

				// create scattering ray
				intersection.ray.origin = intersection.point;
				intersection.ray.direction = sctr_direction;
				intersection.ray.ResetRange();

				return intersection.metalness;
			}
		};
	}
}

#endif // !CUDA_MATERIAL_H