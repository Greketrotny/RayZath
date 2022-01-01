#ifndef MATERIAL_H
#define MATERIAL_H

#include "world_object.h"
#include "render_parts.h"

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
		Graphics::Color color;

		float metalness;
		float roughness;
		float emission;

		float ior;
		float scattering;

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
	};

	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Gold>()
	{
		return ConStruct<Material>(
			"generated_gold",
			Graphics::Color(0xFF, 0xD7, 0x00, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Silver>()
	{
		return ConStruct<Material>(
			"generated_silver",
			Graphics::Color(0xC0, 0xC0, 0xC0, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Copper>()
	{
		return ConStruct<Material>(
			"generated_copper",
			Graphics::Color(0xB8, 0x73, 0x33, 0xFF),
			1.0f, 0.001f, 0.0f, 1.0f, 0.0f);
	}

	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Glass>()
	{
		return ConStruct<Material>(
			"generated_glass",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.45f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Water>()
	{
		return ConStruct<Material>(
			"generated_water",
			Graphics::Color(0xFF, 0xFF, 0xFF, 0x00),
			0.0f, 0.0f, 0.0f, 1.33f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Mirror>()
	{
		return ConStruct<Material>(
			"generated_mirror",
			Graphics::Color(0xF0, 0xF0, 0xF0, 0xFF),
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}

	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::RoughWood>()
	{
		return ConStruct<Material>(
			"generated_rough_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.1f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::PolishedWood>()
	{
		return ConStruct<Material>(
			"generated_polished_wood",
			Graphics::Color(0x96, 0x6F, 0x33, 0xFF),
			0.0f, 0.002f, 0.0f, 1.0f, 0.0f);
	}

	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Paper>()
	{
		return ConStruct<Material>(
			"generated_paper",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Rubber>()
	{
		return ConStruct<Material>(
			"generated_rubber",
			Graphics::Color::Palette::Black,
			0.0f, 0.3f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::RoughPlastic>()
	{
		return ConStruct<Material>(
			"generated_rough_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.45f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::PolishedPlastic>()
	{
		return ConStruct<Material>(
			"generated_polished_plastic",
			Graphics::Color::Palette::White,
			0.0f, 0.0015f, 0.0f, 1.0f, 0.0f);
	}
	template<> inline ConStruct<Material> Material::GenerateMaterial<Material::Common::Porcelain>()
	{
		return ConStruct<Material>(
			"generated_porcelain",
			Graphics::Color::Palette::White,
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	}
}

#endif // !MATERIAL_H