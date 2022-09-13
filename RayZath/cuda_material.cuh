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

		__host__ void reconstruct(
			const World& hCudaWorld,
			const RayZath::Engine::Handle<RayZath::Engine::Material>& hMaterial,
			cudaStream_t& mirror_stream);


	public:
		__host__ void color(const Graphics::Color& color)
		{
			m_color = color;
		}
		__host__ void texture(const Texture* texture)
		{
			mp_texture = texture;
		}
		__host__ void emission(const float emission)
		{
			m_emission = emission;
		}

		__device__ const ColorF& color() const
		{
			return m_color;
		}
		__device__ ColorF color(const Texcrd texcrd) const
		{
			if (mp_texture) return mp_texture->fetch(texcrd);
			else return color();
		}
		__device__ const ColorF opacityColor() const
		{
			ColorF op_color = color();
			op_color.alpha = 1.0f - op_color.alpha;
			return op_color;
		}
		__device__ const ColorF opacityColor(const Texcrd texcrd) const
		{
			ColorF op_color = color(texcrd);
			op_color.alpha = 1.0f - op_color.alpha;
			return op_color;
		}
		__device__ float metalness() const
		{
			return m_metalness;
		}
		__device__ float metalness(const Texcrd texcrd) const
		{
			if (mp_metalness_map) return mp_metalness_map->fetch(texcrd);
			else return metalness();
		}
		__device__ float roughness() const
		{
			return m_roughness;
		}
		__device__ float roughness(const Texcrd texcrd) const
		{
			if (mp_roughness_map) return mp_roughness_map->fetch(texcrd);
			else return roughness();
		}
		__device__ float emission() const
		{
			return m_emission;
		}
		__device__ float emission(const Texcrd texcrd) const
		{
			if (mp_emission_map) return mp_emission_map->fetch(texcrd);
			else return emission();
		}

		__device__ float ior() const
		{
			return m_ior;
		}
		__device__ float scattering() const
		{
			return m_scattering;
		}

		__device__ const NormalMap* normalMap() const
		{
			return mp_normal_map;
		}


	public:
		__device__ bool applyScattering(
			SceneRay& ray, 
			SurfaceProperties& surface, 
			RNG& rng) const
		{
			if (scattering() > 1.0e-4f)
			{
				const float scatter_distance = (-cui_logf(rng.unsignedUniform() + 1.0e-4f)) / scattering();
				if (scatter_distance < ray.near_far.y)
				{
					ray.near_far.y = scatter_distance;
					surface.surface_material = this;
					surface.behind_material = this;
					surface.normal = surface.mapped_normal = ray.direction;
					return true;
				}
			}
			return false;
		}

		// bidirectional reflection distribution function
		__device__ float BRDF(const RangedRay& ray, const SurfaceProperties& surface, const vec3f& vPL) const
		{
			if (surface.surface_material->scattering() > 0.0f) return 1.0f;

			const float vN_dot_vO = vec3f::dotProduct(surface.mapped_normal, vPL);
			if (vN_dot_vO <= 0.0f) return 0.0f;
			const float vN_dot_vI = vec3f::dotProduct(surface.mapped_normal, -ray.direction);
			if (vN_dot_vI <= 0.0f) return 0.0f;

			const vec3f vI_half_vO = halfwayVector(ray.direction, vPL);

			const float nornal_distribution = NDF(surface.mapped_normal, vI_half_vO, surface.roughness);
			const float atten_i = attenuation(vN_dot_vI, surface.roughness);
			const float atten_o = attenuation(vN_dot_vO, surface.roughness);
			const float attenuation = atten_i * atten_o;

			const float diffuse = vN_dot_vO * float(surface.color.alpha == 0.0f);
			const float specular = nornal_distribution * attenuation / (vN_dot_vI * vN_dot_vO);

			return lerp(diffuse, specular * vN_dot_vO, surface.reflectance);
		}
		__device__ ColorF BRDFColor(const SurfaceProperties& surface) const
		{
			return lerp(surface.color, ColorF(1.0f), surface.reflectance);
		}
	private:
		// normal distribution function
		__device__ float NDF(const vec3f vN, const vec3f vH, const float roughness) const
		{
			const float vN_dot_vH = vec3f::dotProduct(vN, vH);
			const float b = (vN_dot_vH * vN_dot_vH) * (roughness - 1.0f) + 1.0001f;
			return (roughness + 1.0e-5f) / (b * b);
		}
		__device__ float attenuation(const float cos_angle, const float roughness) const
		{
			return cos_angle / ((cos_angle * (1.0f - roughness)) + roughness);
		}


		// ray generation
	public:
		__device__ vec3f sampleDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
		{
			if (surface.color.alpha > 0.0f)
			{	// transmission
				if (surface.surface_material->m_scattering > 0.0f)
				{
					return sampleScatteringDirection(ray, surface, rng);
				}
				else
				{
					return sampleTransmissionDirection(ray, surface, rng);
				}
			}
			else
			{	// reflection

				if (rng.unsignedUniform() > surface.reflectance)
				{	// diffuse reflection
					return sampleDiffuseDirection(ray, surface, rng);
				}
				else
				{	// glossy reflection
					return sampleGlossyDirection(ray, surface, rng);
				}
			}
		}
	private:
		__device__ vec3f sampleDiffuseDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
		{
			vec3f vO = cosineSampleHemisphere(
				rng.unsignedUniform(),
				rng.unsignedUniform(),
				surface.mapped_normal);

			// flip sample above surface if needed
			const float vR_dot_vN = vec3f::Similarity(vO, surface.normal);
			if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

			surface.tint_factor = 1.0f;
			return vO;
		}
		__device__ vec3f sampleGlossyDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
		{
			const vec3f vH = sampleHemisphere(
				rng.unsignedUniform(),
				1.0f - cui_powf(
					rng.unsignedUniform() + 1.0e-5f,
					surface.roughness),
				surface.mapped_normal);

			// calculate reflection direction
			vec3f vO = reflectVector(ray.direction, vH);

			// reflect sample above surface if needed
			const float vR_dot_vN = vec3f::Similarity(vO, surface.normal);
			if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

			surface.tint_factor = surface.metalness;
			return vO;
		}
		__device__ vec3f sampleTransmissionDirection(SceneRay& ray, SurfaceProperties& surface, RNG& rng) const
		{
			if (surface.fresnel < rng.unsignedUniform())
			{	// transmission

				const vec3f vO = ray.direction * surface.refraction_factors.x +
					surface.mapped_normal * surface.refraction_factors.y;

				ray.material = surface.behind_material;
				surface.normal.Reverse();
				surface.tint_factor = surface.metalness;
				return vO;
			}
			else
			{	// reflection

				vec3f vO = reflectVector(ray.direction,	surface.mapped_normal);

				// flip sample above surface if needed
				const float vR_dot_vN = vec3f::dotProduct(vO, surface.normal);
				if (vR_dot_vN < 0.0f) vO += surface.normal * -2.0f * vR_dot_vN;

				surface.tint_factor = 1.0f;
				return vO;
			}
		}
		__device__ vec3f sampleScatteringDirection(
			SceneRay& ray, SurfaceProperties& surface,
			RNG& rng) const
		{
			// generate scatter direction
			const vec3f vO = sampleSphere(
				rng.unsignedUniform(),
				rng.unsignedUniform(),
				ray.direction);

			surface.tint_factor = surface.metalness;
			return vO;
		}
	};
}

#endif // !CUDA_MATERIAL_H