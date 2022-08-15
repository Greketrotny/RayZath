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
		void SetColor(const Graphics::Color& color);
		void SetMetalness(const float& metalness);
		void SetRoughness(const float& roughness);
		void SetEmission(const float& emission);
		void SetIOR(const float& ior);
		void SetScattering(const float& scattering);

		void SetTexture(const Handle<Texture>& texture);
		void SetNormalMap(const Handle<NormalMap>& normal_map);
		void SetMetalnessMap(const Handle<MetalnessMap>& metalness_map);
		void SetRoughnessMap(const Handle<RoughnessMap>& roughness_map);
		void SetEmissionMap(const Handle<EmissionMap>& emission_map);


		const Graphics::Color& GetColor() const noexcept;
		float GetMetalness() const noexcept;
		float GetRoughness() const noexcept;
		float GetEmission() const noexcept;
		float GetIOR() const noexcept;
		float GetScattering() const noexcept;

		const Handle<Texture>& GetTexture() const;
		const Handle<NormalMap>& GetNormalMap() const;
		const Handle<MetalnessMap>& GetMetalnessMap() const;
		const Handle<RoughnessMap>& GetRoughnessMap() const;
		const Handle<EmissionMap>& GetEmissionMap() const;
	private:
		void ResourceNotify();

	public:
		template <Common M>
		static ConStruct<Material> GenerateMaterial();

		bool LoadFromFile(const std::string& file_name);
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

			name = material->GetName();

			color = material->GetColor();
			metalness = material->GetMetalness();
			roughness = material->GetRoughness();
			emission = material->GetEmission();
			ior = material->GetIOR();
			scattering = material->GetScattering();

			texture = material->GetTexture();
			normal_map = material->GetNormalMap();
			metalness_map = material->GetMetalnessMap();
			roughness_map = material->GetRoughnessMap();
			emission_map = material->GetEmissionMap();
		}
	};

	
}

#endif // !MATERIAL_H