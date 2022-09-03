#ifndef JSON_SAVER_H
#define JSON_SAVER_H

#include "world.hpp"
#include "saver.hpp"

#include "./lib/Json/json.hpp"

#include <fstream>
#include <filesystem>

namespace RayZath::Engine
{
	/// <summary>
	/// Stores names of objects already saved. Provides unique naming for later reference in higher order objects.
	/// </summary>
	template <World::ObjectType... Ts>
	struct ObjectNames
	{
	public:
		template <World::ObjectType T>
		static constexpr bool is_with_path = Utils::is::value<T>::template any_of<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap,
			World::ObjectType::Material,
			World::ObjectType::MeshStructure>::value;
		template <World::ObjectType T, typename = void>
		struct value_t
		{
			using type = std::tuple<std::string>;
		};
		template <World::ObjectType T>
		struct value_t<T, std::enable_if_t<is_with_path<T>>>
		{
			using type = std::tuple<std::string, std::filesystem::path>;
		};

		template <World::ObjectType T>
		using object_name_map_t = std::unordered_map<Handle<World::object_t<T>>, typename value_t<T>::type>;
		template <World::ObjectType T>
		using name_set_t = std::unordered_set<std::string_view>;
		template <World::ObjectType T>
		using type_names_t = std::pair<object_name_map_t<T>, name_set_t<T>>;
		using names_t = std::tuple<type_names_t<Ts>...>;
	private:
		names_t m_names;

	public:
		template <World::ObjectType T>
		bool contains(const Handle<World::object_t<T>>& object)
		{
			const auto& [map, set] = get<T>();
			return map.find(object) != map.end();
		}
		template <World::ObjectType T>
		std::string uniqueName(const std::string& name)
		{
			const auto& [map, set] = get<T>();

			uint32_t counter = 1;
			std::stringstream unique_name{name};
			while (set.count(unique_name.str()) == 1)
				unique_name = std::stringstream{} << name << "." << std::setw(3) << std::setfill('0') << counter++;
			return unique_name.str();
		}
		template <World::ObjectType T, std::enable_if_t<!is_with_path<T>, bool> = true>
		void add(const Handle<World::object_t<T>>& object, std::string unique_name)
		{
			auto& [map, set] = get<T>();
			[[maybe_unused]] const auto& [inserted_object, object_inserted] = map.insert({object, {std::move(unique_name)}});
			RZAssertCore(object_inserted, "the same object of the same type inserted twice");
			[[maybe_unused]] const auto& [_, name_inserted] =
				set.insert(
					std::string_view{std::get<0>(inserted_object->second).data(), 
					std::get<0>(inserted_object->second).size()});
			RZAssertCore(name_inserted, "same name inserted twice");
		}
		template <World::ObjectType T, std::enable_if_t<is_with_path<T>, bool> = true>
		void add(
			const Handle<World::object_t<T>>& object, 
			std::string unique_name,
			std::filesystem::path path)
		{
			auto& [map, set] = get<T>();
			[[maybe_unused]] const auto& [inserted_object, object_inserted] = map.insert(
				{object, {std::move(unique_name), std::move(path)}});
			RZAssertCore(object_inserted, "the same object of the same type inserted twice");
			[[maybe_unused]] const auto& [_, name_inserted] =
				set.insert(
					std::string_view{std::get<0>(inserted_object->second).data(),
					std::get<0>(inserted_object->second).size()});
			RZAssertCore(name_inserted, "same name inserted twice");
		}
		template <World::ObjectType T>
		const auto& name(const Handle<World::object_t<T>>& object)
		{
			return std::get<0>(get<T>().first.at(object));
		}
		template <World::ObjectType T>
		const auto& path(const Handle<World::object_t<T>>& object)
		{
			static_assert(is_with_path<T>);
			return std::get<1>(get<T>().first.at(object));
		}
		void reset()
		{
			(..., (get<Ts>().first.clear(), get<Ts>().second.clear()));
		}

	private:
		template <World::ObjectType T>
		auto& get()
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_names);
		}
		template <World::ObjectType T>
		const auto& get() const
		{
			return std::get<Utils::index_of::value<T>::template in_sequence<Ts...>>(m_names);
		}
	};

	class JsonSaver
	{
	public:
		using json_t = nlohmann::json;
	private:
		World& mr_world;
		std::filesystem::path m_path; // path to scene (without file_name.json)
		json_t m_json;

		ObjectNames<
			World::ObjectType::Texture,
			World::ObjectType::NormalMap,
			World::ObjectType::MetalnessMap,
			World::ObjectType::RoughnessMap,
			World::ObjectType::EmissionMap,
			World::ObjectType::Material,
			World::ObjectType::MeshStructure,
			World::ObjectType::Camera,
			World::ObjectType::SpotLight,
			World::ObjectType::DirectLight,
			World::ObjectType::Mesh,
			World::ObjectType::Group> m_names;

		struct Paths
		{
		private:
			static constexpr char
				textures[] = "maps\\texture\\",
				normal_maps[] = "maps\\normal\\",
				metalness_maps[] = "maps\\metalness\\",
				roughness_maps[] = "maps\\roughness\\",
				emission_maps[] = "maps\\emission\\",
				materials[] = "materials\\",
				meshes[] = "meshes\\";
		public:
			template <World::ObjectType T>
			static constexpr auto path = Utils::static_dictionary::vv_translate<T>::template with<
				Utils::static_dictionary::vv_translation<World::ObjectType::Texture, textures>,
				Utils::static_dictionary::vv_translation<World::ObjectType::NormalMap, normal_maps>,
				Utils::static_dictionary::vv_translation<World::ObjectType::MetalnessMap, metalness_maps>, 
				Utils::static_dictionary::vv_translation<World::ObjectType::RoughnessMap, roughness_maps>, 
				Utils::static_dictionary::vv_translation<World::ObjectType::EmissionMap, emission_maps>, 
				Utils::static_dictionary::vv_translation<World::ObjectType::Material, materials>, 
				Utils::static_dictionary::vv_translation<World::ObjectType::MeshStructure, meshes>>::value;
		};

	public:
		JsonSaver(World& world);

	public:
		void saveJsonScene(const Saver::SaveOptions& options);
	private:
		template <World::ObjectType T>
		void save(json_t& json);

		template <World::ObjectType T>
		void saveMap(const std::string& map_key, json_t& json);

		void saveSpecialMaterial(const char* key, const Material& material, json_t& json);
	};
}

#endif
