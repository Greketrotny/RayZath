#ifndef MATERIAL_H
#define MATERIAL_H

#include "world_object.h"
#include "render_parts.h"

namespace RayZath
{
	struct Material
		: public WorldObject
	{
	private:
		Graphics::Color m_color;

		float m_metalness;
		float m_specularity;
		float m_roughness;
		float m_emission;

		float m_ior;
		float m_scattering;

		Observer<Texture> m_texture;
		Observer<NormalMap> m_normal_map;
		Observer<MetalnessMap> m_metalness_map;
		Observer<SpecularityMap> m_specularity_map;
		Observer<RoughnessMap> m_roughness_map;
		Observer<EmissionMap> m_emission_map;


	public:
		Material(
			Updatable* updatable, 
			const ConStruct<Material>& con_struct);
		Material(const Material& material) = delete;
		Material(Material&& material) = delete;


	public:
		Material& operator=(const Material& material) = delete;
		Material& operator=(Material&& material) = delete;


	public:
		void SetColor(const Graphics::Color& color);
		void SetMetalness(const float& metalness);
		void SetSpecularity(const float& specularity);
		void SetRoughness(const float& roughness);
		void SetEmission(const float& emission);
		void SetIOR(const float& ior);
		void SetScattering(const float& scattering);

		void SetTexture(const Handle<Texture>& texture);
		void SetNormalMap(const Handle<NormalMap>& normal_map);
		void SetMetalnessMap(const Handle<MetalnessMap>& metalness_map);
		void SetSpecularityMap(const Handle<SpecularityMap>& specularity_map);
		void SetRoughnessMap(const Handle<RoughnessMap>& roughness_map);
		void SetEmissionMap(const Handle<EmissionMap>& emission_map);


		const Graphics::Color& GetColor() const noexcept;
		float GetMetalness() const noexcept;
		float GetSpecularity() const noexcept;
		float GetRoughness() const noexcept;
		float GetEmission() const noexcept;
		float GetIOR() const noexcept;
		float GetScattering() const noexcept;

		const Handle<Texture>& GetTexture() const;
		const Handle<NormalMap>& GetNormalMap() const;
		const Handle<MetalnessMap>& GetMetalnessMap() const;
		const Handle<SpecularityMap>& GetSpecularityMap() const;
		const Handle<RoughnessMap>& GetRoughnessMap() const;
		const Handle<EmissionMap>& GetEmissionMap() const;
	private:
		void ResourceNotify();
	};

	template<> struct ConStruct<Material> : public ConStruct<WorldObject>
	{
		Graphics::Color color;

		float metalness;
		float specularity;
		float roughness;
		float emission;

		float ior;
		float scattering;

		Handle<Texture> texture;
		Handle<NormalMap> normal_map;
		Handle<MetalnessMap> metalness_map;
		Handle<SpecularityMap> specularity_map;
		Handle<RoughnessMap> roughness_map;
		Handle<EmissionMap> emission_map;

		ConStruct(
			const Graphics::Color& color = Graphics::Color::Palette::LightGrey,
			const float& metalness = 0.0f,
			const float& specularity = 0.0f,
			const float& roughness = 0.0f,
			const float& emission = 0.0f,
			const float& ior = 1.0f,
			const float& scattering = 0.0f,
			const Handle<Texture>& texture = Handle<Texture>(),
			const Handle<NormalMap>& normal_map = Handle<NormalMap>(),
			const Handle<MetalnessMap>& metalness_map = Handle<MetalnessMap>(),
			const Handle<SpecularityMap>& specularity_map = Handle<SpecularityMap>(),
			const Handle<RoughnessMap>& roughness_map = Handle<RoughnessMap>(),
			const Handle<EmissionMap>& emission_map = Handle<EmissionMap>())
			: color(color)
			, metalness(metalness)
			, specularity(specularity)
			, roughness(roughness)
			, emission(emission)
			, ior(ior)
			, scattering(scattering)
			, texture(texture)
			, normal_map(normal_map)
			, metalness_map(metalness_map)
			, specularity_map(specularity_map)
			, roughness_map(roughness_map)
			, emission_map(emission_map)
		{}
	};
}

#endif // !MATERIAL_H