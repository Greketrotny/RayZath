#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "cuda_buffer.cuh"
#include "cuda_render_parts.cuh"

namespace RayZath::Cuda
{
	struct Material
	{
	private:
		ColorF m_color;

		float m_metalness;
		float m_roughness;
		float m_emission;

		float m_ior;
		float m_scattering;

		const Texture* mp_texture;
		const NormalMap* mp_normal_map;
		const MetalnessMap* mp_metalness_map;
		const RoughnessMap* mp_roughness_map;
		const EmissionMap* mp_emission_map;

	public:
		__host__ __device__ Material(
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

		__host__ Material& operator=(const RayZath::Engine::Material& hMaterial);

		__host__ void Reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Material>& hMaterial,
			cudaStream_t& mirror_stream);


	public:
		__host__ void SetColor(const Graphics::Color& color)
		{
			m_color = color;
		}
		__host__ void SetTexture(const Texture* texture)
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
		__device__ ColorF GetColor(const Texcrd texcrd) const
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
		__device__ const ColorF GetOpacityColor(const Texcrd texcrd) const
		{
			ColorF color = GetColor(texcrd);
			color.alpha = 1.0f - color.alpha;
			return color;
		}
		__device__ float GetMetalness() const
		{
			return m_metalness;
		}
		__device__ float GetMetalness(const Texcrd texcrd) const
		{
			if (mp_metalness_map) return mp_metalness_map->Fetch(texcrd);
			else return GetMetalness();
		}
		__device__ float GetRoughness() const
		{
			return m_roughness;
		}
		__device__ float GetRoughness(const Texcrd texcrd) const
		{
			if (mp_roughness_map) return mp_roughness_map->Fetch(texcrd);
			else return GetRoughness();
		}
		__device__ float GetEmission() const
		{
			return m_emission;
		}
		__device__ float GetEmission(const Texcrd texcrd) const
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

		__device__ const NormalMap* GetNormalMap() const
		{
			return mp_normal_map;
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
					intersection.surface_normal = intersection.ray.direction;
					intersection.mapped_normal = intersection.surface_normal;
					return true;
				}
			}
			return false;
		}
		__device__ bool SampleDirect(
			const RayIntersection& intersection) const
		{
			return 
				intersection.surface_material->GetScattering() > 1.0e-4f || 
				intersection.color.alpha == 0.0f;
		}

	public:
		// bidirectional reflection distribution function
		__device__ float BRDF(
			const RayIntersection& intersection,
			const vec3f& vPL) const
		{
			if (intersection.surface_material->GetScattering() > 0.0f) return 1.0f;

			const float vN_dot_vO = vec3f::DotProduct(intersection.mapped_normal, vPL);
			if (vN_dot_vO <= 0.0f) return 0.0f;

			const float vN_dot_vI = vec3f::DotProduct(intersection.mapped_normal, -intersection.ray.direction);
			const vec3f vI_half_vO = HalfwayVector(intersection.ray.direction, vPL);

			const float nornal_distribution = NDF(
				intersection.mapped_normal,
				vI_half_vO,
				intersection.roughness);
			const float atten_i = Attenuation(vN_dot_vI, intersection.roughness);
			const float atten_o = Attenuation(vN_dot_vO, intersection.roughness);
			const float attenuation = atten_i * atten_o;

			const float diffuse = vN_dot_vO;
			const float specular = nornal_distribution * attenuation / (vN_dot_vI * vN_dot_vO);

			return Lerp(diffuse, specular * vN_dot_vO, intersection.reflectance);
		}
		__device__ ColorF BRDFColor(const RayIntersection& intersection) const
		{
			return Lerp(intersection.color, ColorF(1.0f), intersection.reflectance);
		}
	private:
		// normal distribution function
		__device__ float NDF(
			const vec3f vN, const vec3f vH,
			const float roughness) const
		{
			const float vN_dot_vH = vec3f::DotProduct(vN, vH);
			const float b = (vN_dot_vH * vN_dot_vH) * (roughness - 1.0f) + 1.0001f;
			return (roughness + 1.0e-5f) / (b * b);
		}
		__device__ float Attenuation(
			const float cos_angle,
			const float roughness) const
		{
			return cos_angle / ((cos_angle * (1.0f - roughness)) + roughness);
		}


		// ray generation
	public:
		__device__ vec3f SampleDirection(
			RayIntersection& intersection,
			RNG& rng) const
		{
			if (intersection.color.alpha > 0.0f)
			{	// transmission
				if (intersection.surface_material->m_scattering > 0.0f)
				{
					return SampleScatteringDirection(intersection, rng);
				}
				else
				{
					return SampleTransmissionDirection(intersection, rng);
				}
			}
			else
			{	// reflection

				if (rng.UnsignedUniform() > intersection.reflectance)
				{	// diffuse reflection
					return SampleDiffuseDirection(intersection, rng);
				}
				else
				{	// glossy reflection
					return SampleGlossyDirection(intersection, rng);
				}
			}
		}
	private:
		__device__ vec3f SampleDiffuseDirection(
			RayIntersection& intersection,
			RNG& rng) const
		{
			vec3f vO = CosineSampleHemisphere(
				rng.UnsignedUniform(),
				rng.UnsignedUniform(),
				intersection.mapped_normal);

			// flip sample above surface if needed
			const float vR_dot_vN = vec3f::Similarity(vO, intersection.surface_normal);
			if (vR_dot_vN < 0.0f) vO += intersection.surface_normal * -2.0f * vR_dot_vN;

			intersection.next_ray_metalness = 1.0f;
			return vO;
		}
		__device__ vec3f SampleGlossyDirection(
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
			vec3f vO = ReflectVector(
				intersection.ray.direction,
				vH);

			// reflect sample above surface if needed
			const float vR_dot_vN = vec3f::Similarity(vO, intersection.surface_normal);
			if (vR_dot_vN < 0.0f) vO += intersection.surface_normal * -2.0f * vR_dot_vN;

			intersection.next_ray_metalness = intersection.metalness;
			return vO;
		}
		__device__ vec3f SampleTransmissionDirection(
			RayIntersection& intersection,
			RNG& rng) const
		{
			if (intersection.fresnel < rng.UnsignedUniform())
			{	// transmission

				const float n1 = intersection.ray.material->m_ior;
				const float n2 = intersection.behind_material->m_ior;
				const float ratio = n1 / n2;
				const float cosi = fabsf(vec3f::DotProduct(
					intersection.ray.direction, intersection.mapped_normal));
				const float sin2_t = ratio * ratio * (1.0f - cosi * cosi);
				const float cost = sqrtf(1.0f - sin2_t);

				const vec3f vO = intersection.ray.direction * ratio +
					intersection.mapped_normal * (ratio * cosi - cost);

				intersection.ray.material = intersection.behind_material;
				intersection.next_ray_metalness = intersection.metalness;
				intersection.surface_normal.Reverse();
				return vO;
			}
			else
			{	// reflection

				vec3f vO = ReflectVector(
					intersection.ray.direction,
					intersection.mapped_normal);

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::DotProduct(vO, intersection.surface_normal);
				if (vR_dot_vN < 0.0f) vO += intersection.surface_normal * -2.0f * vR_dot_vN;

				intersection.next_ray_metalness = 1.0f;
				return vO;
			}
		}
		__device__ vec3f SampleScatteringDirection(
			RayIntersection& intersection,
			RNG& rng) const
		{
			// generate scatter direction
			const vec3f vO = SampleSphere(
				rng.UnsignedUniform(),
				rng.UnsignedUniform(),
				intersection.ray.direction);

			intersection.next_ray_metalness = intersection.metalness;
			return vO;
		}
	};
}

#endif // !CUDA_MATERIAL_H