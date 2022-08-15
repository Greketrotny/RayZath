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
	private:
		World& mr_world;
		std::filesystem::path m_path;

	public:
		JsonLoader(World& world);

	public:
		void LoadJsonScene(std::ifstream& file, const std::filesystem::path& path);
	private:
		std::filesystem::path ModifyPath(std::filesystem::path path);

		void LoadMaterial(const json_t& json, Material& material);

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
