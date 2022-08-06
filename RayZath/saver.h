#ifndef SAVER_H
#define SAVER_H

#include "world.h"
#include "index_of.h"

#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace RayZath::Engine
{
	/// <summary>
	/// Stores names of objects already saved. Provides unique naming for later reference in higher order objects.
	/// </summary>
	/*template <World::ObjectType... Ts>
	struct ObjectNames
	{
	public:
		template <World::ObjectType T>
		using object_name_map_t = std::unordered_map<Handle<World::object_t<T>>, std::string>;
		template <World::ObjectType T>
		using name_set_t = std::unordered_set<std::string_view>;
		template <World::ObjectType T>
		using type_names_t = std::pair<object_name_map_t<T>, name_set_t<T>>;
		using names_t = std::tuple<type_names_t<Ts>...>;
	private:
		names_t m_names;

	public:
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

		template <World::ObjectType T>
		void add(const Handle<World::object_t<T>>& object, std::string unique_name)
		{
			auto& [map, set] = get<T>();
			[[maybe_unused]] const auto& [inserted_object, object_inserted] = map.insert({object, std::move(unique_name)});
			RZAssert(object_inserted, "the same object of the same type inserted twice");
			[[maybe_unused]] const auto& [_, name_inserted] = 
				set.insert(std::string_view{inserted_object->second.data(), inserted_object->second.size()});
			RZAssert(name_inserted, "same name inserted twice");
		}
		template <World::ObjectType T>
		const auto& name(const Handle<World::object_t<T>>& object)
		{
			return get<T>().first.at(object);
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
	};*/

	class SaverBase
	{
	protected:
		World& mr_world;
		/*ObjectNames< 
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
			World::ObjectType::Group> m_names;*/

	public:
		SaverBase(World& world)
			: mr_world(world)
		{}
	};
	class BitmapSaver : public SaverBase
	{
	public:
		BitmapSaver(World& world)
			: SaverBase(world)
		{}


		/*void SaveAllMaps(const std::filesystem::path& path);
		template <World::ObjectType T>
		void SaveAllTypeMaps(const std::filesystem::path& path)
		{
			auto& maps = mr_world.Container<T>();
			if (maps.GetCount() == 0) return;
			std::filesystem::create_directory(path);
			for (uint32_t id = 0; id < maps.GetCount(); id++)
			{
				if (auto& map = maps[id]; map)
				{
					auto unique_name{m_names.uniqueName<T>(map->GetName())};
					auto full_path = SaveMap<T>(map->GetBitmap(), path, unique_name);
					m_names.add<T>(map, std::move(full_path.string()));
				}
			}
		}*/
		template <World::ObjectType T>
		std::filesystem::path SaveMap(
			const typename World::object_t<T>::buffer_t& map,
			const std::filesystem::path& path,
			const std::string& file_name);
	};
	class MTLSaver : public BitmapSaver
	{
	public:
		struct MapsPaths
		{
			const std::filesystem::path& texture, &normal, &metalness, &roughness, &emission;
		};

		MTLSaver(World& world)
			: BitmapSaver(world)
		{}

		void SaveMTL(
			const Handle<Material>& material, 
			const std::filesystem::path& path,
			const std::optional<MapsPaths>& maps_paths,
			const std::string& file_name);
		void SaveMTLWithMaps(
			const Handle<Material>& material,
			const std::filesystem::path& path,
			const std::string& file_name);
	};
	class OBJSaver : MTLSaver
	{
	public:
		OBJSaver(World& world)
			: MTLSaver(world)
		{}

		void SaveOBJ(
			const std::vector<Handle<MeshStructure>>& meshes,
			const std::filesystem::path& path,
			const std::filesystem::path& material_library,
			const std::unordered_map<uint32_t, std::string>& material_names);
	private:
		void SaveMesh(
			const Handle<MeshStructure>& mesh,
			std::ofstream& file,
			const std::unordered_map<uint32_t, std::string>& material_names);
	};
	class Saver : public OBJSaver
	{
	public:
		struct SaveOptions
		{
			std::filesystem::path path{};
			bool allow_partial_write = true; // when saving scene fails, no saved content is removed
			bool duplicate_textures = false; // when two materials reference the same texture, will be saved twice
			bool duplicate_materials = false; // hwen two instances reference the same material, will be saved twice
			bool group_materials_for_object = false; // a few materials can be saved in one .mtl file
		};

	public:
		Saver(World& world)
			: OBJSaver(world)
		{}

	public:
		void SaveScene(const SaveOptions& options);
	};
}

#endif //!LOADER_H
