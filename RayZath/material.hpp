#ifndef MATERIAL_H
#define MATERIAL_H

#include "world_object.hpp"
#include "render_parts.hpp"

namespace RayZath::Engine
{
	struct Material;
	template <> struct ConStruct<Material>;

	struct Material
		: public WorldObject
	{
	public:
		enum class Common
		{
			Gold,
			Silver,
			Copper,

			Glass,
			Water,
			Mirror,

			RoughWood,
			PolishedWood,
			Paper,
			Rubber,
			RoughPlastic,
			PolishedPlastic,
			Porcelain
		};
	private:
		Graphics::Color m_color;

		float m_metalness;
		float m_roughness;
		float m_emission;

		float m_ior;
		float m_scattering;

		Observer<Texture> m_texture;
		Observer<NormalMap> m_normal_map;
		Observer<MetalnessMap> m_metalness_map;
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
		void color(const Graphics::Color& color);
		void metalness(const float& metalness);
		void roughness(const float& roughness);
		void emission(const float& emission);
		void ior(const float& ior);
		void scattering(const float& scattering);

		void texture(const Handle<Texture>& texture);
		void normalMap(const Handle<NormalMap>& normal_map);
		void metalnessMap(const Handle<MetalnessMap>& metalness_map);
		void roughnessMap(const Handle<RoughnessMap>& roughness_map);
		void emissionMap(const Handle<EmissionMap>& emission_map);

		const Graphics::Color& color() const noexcept;
		float metalness() const noexcept;
		float roughness() const noexcept;
		float emission() const noexcept;
		float ior() const noexcept;
		float scattering() const noexcept;

		const Handle<Texture>& texture() const;
		const Handle<NormalMap>& normalMap() const;
		const Handle<MetalnessMap>& metalnessMap() const;
		const Handle<RoughnessMap>& roughnessMap() const;
		const Handle<EmissionMap>& emissionMap() const;
	private:
		void resourceNotify();

	public:
		template <Common M>
		static ConStruct<Material> generateMaterial();
	};

	template<> 
	struct ConStruct<Material>
		: public ConStruct<WorldObject>
	{
		Graphics::Color color = Graphics::Color::Palette::LightGrey;

		float metalness = 0.0f;
		float roughness = 0.0f;
		float emission = 0.0f;
		float ior = 1.0f;
		float scattering = 0.0f;

		Handle<Texture> texture;
		Handle<NormalMap> normal_map;
		Handle<MetalnessMap> metalness_map;
		Handle<RoughnessMap> roughness_map;
		Handle<EmissionMap> emission_map;

		ConStruct(
			const std::string& name = "material name",
			const Graphics::Color& color = Graphics::Color::Palette::LightGrey,
			const float& metalness = 0.0f,
			const float& roughness = 0.0f,
			const float& emission = 0.0f,
			const float& ior = 1.5f,
			const float& scattering = 0.0f,
			const Handle<Texture>& texture = Handle<Texture>(),
			const Handle<NormalMap>& normal_map = Handle<NormalMap>(),
			const Handle<MetalnessMap>& metalness_map = Handle<MetalnessMap>(),
			const Handle<RoughnessMap>& roughness_map = Handle<RoughnessMap>(),
			const Handle<EmissionMap>& emission_map = Handle<EmissionMap>())
			: ConStruct<WorldObject>(name)
			, color(color)
			, metalness(metalness)
			, roughness(roughness)
			, emission(emission)
			, ior(ior)
			, scattering(scattering)
			, texture(texture)
			, normal_map(normal_map)
			, metalness_map(metalness_map)
			, roughness_map(roughness_map)
			, emission_map(emission_map)
		{}
		ConStruct(const Handle<Material>& material)
		{
			if (!material) return;

			name = material->name();

			color = material->color();
			metalness = material->metalness();
			roughness = material->roughness();
			emission = material->emission();
			ior = material->ior();
			scattering = material->scattering();

			texture = material->texture();
			normal_map = material->normalMap();
			metalness_map = material->metalnessMap();
			roughness_map = material->roughnessMap();
			emission_map = material->emissionMap();
		}
	};

	
}

#endif // !MATERIAL_H