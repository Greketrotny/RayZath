#ifndef MATERIAL_H
#define MATERIAL_H

#include "world_object.hpp"
#include "texture_buffer.hpp"
#include "render_parts.hpp"
#include "typedefs.hpp"

namespace RayZath::Engine
{
	class Material;
	template <> struct ConStruct<Material>;

	class Material
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

		template <ObjectType T>
		static constexpr std::size_t s_map_idx = Utils::static_dictionary::vv_translate<T>::template with<
			Utils::static_dictionary::vv_translation<ObjectType::Texture, 0>,
			Utils::static_dictionary::vv_translation<ObjectType::NormalMap, 1>,
			Utils::static_dictionary::vv_translation<ObjectType::MetalnessMap, 2>,
			Utils::static_dictionary::vv_translation<ObjectType::RoughnessMap, 3>,
			Utils::static_dictionary::vv_translation<ObjectType::EmissionMap, 4>>::value;
		template <ObjectType T>
		using map_t = typename Utils::static_dictionary::vt_translate<T>::template with<
			Utils::static_dictionary::vt_translation<ObjectType::Texture, Texture>,
			Utils::static_dictionary::vt_translation<ObjectType::NormalMap, NormalMap>,
			Utils::static_dictionary::vt_translation<ObjectType::MetalnessMap, MetalnessMap>,
			Utils::static_dictionary::vt_translation<ObjectType::RoughnessMap, RoughnessMap>,
			Utils::static_dictionary::vt_translation<ObjectType::EmissionMap, EmissionMap>>::value;
		std::tuple<
			Observer<Texture>,
			Observer<NormalMap>,
			Observer<MetalnessMap>,
			Observer<RoughnessMap>,
			Observer<EmissionMap>> m_maps;

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

		const Graphics::Color& color() const noexcept;
		float metalness() const noexcept;
		float roughness() const noexcept;
		float emission() const noexcept;
		float ior() const noexcept;
		float scattering() const noexcept;

	private:
		template <ObjectType T>
		auto& mapGetter() { return std::get<s_map_idx<T>>(m_maps); }
		template <ObjectType T>
		const auto& mapGetter() const { return std::get<s_map_idx<T>>(m_maps); }
	public:
		template <ObjectType T>
		decltype(auto) map() const
		{
			return static_cast<const Handle<map_t<T>>&>(mapGetter<T>());
		}
		template <ObjectType T>
		void map(const Handle<map_t<T>>& map)
		{
			mapGetter<T>() = map;
			stateRegister().MakeModified();
		}
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
						
			texture = material->map<ObjectType::Texture>();
			normal_map = material->map<ObjectType::NormalMap>();
			metalness_map = material->map<ObjectType::MetalnessMap>();
			roughness_map = material->map<ObjectType::RoughnessMap>();
			emission_map = material->map<ObjectType::EmissionMap>();
		}
	};

	
}

#endif // !MATERIAL_H