#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include "loader.hpp"

#include "world.hpp"

#include "./lib/Json/json.hpp"

#include <fstream>
#include <filesystem>

namespace RayZath::Engine
{
	class JsonLoader
	{
	public:
		using json_t = nlohmann::json;
		using loaded_set_t = World::apply_all_types<LoadedSet>::type;
		using loaded_set_view_t = World::apply_all_types<LoadedSetView>::type;
	private:
		std::reference_wrapper<World> mr_world;
		std::filesystem::path m_path;	// path to .json scene file (with file name)
		LoadResult m_load_result;
		loaded_set_t m_loaded_set;
		loaded_set_view_t m_loaded_set_view;

	public:
		JsonLoader(std::reference_wrapper<World> world, std::filesystem::path file_path);

		LoadResult load();
	private:
		std::filesystem::path makeLoadPath(std::filesystem::path path);
		std::filesystem::path makeLoadPath(std::filesystem::path path, std::filesystem::path base);
				
		void LoadMaterial(const json_t& json, Material& material);
		void doLoadMaterial(const json_t& json, Material& material);
		void generateMaterial(const json_t& json, Material& material);
		Handle<MeshStructure> generateMesh(const json_t& json);

		template <World::ObjectType T, typename U = World::object_t<T>, typename = void>
		Handle<U> Load(const json_t& object_json);
		template <World::ObjectType T, typename U = World::object_t<T>>
		Handle<U> LoadMap(const json_t& object_json);

		template <World::ObjectType T, typename U = World::object_t<T>>
		void ObjectLoad(const json_t& world_json, const std::string& key);
		void LoadWorld(const json_t& world_json);
	};
}

#endif
