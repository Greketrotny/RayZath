#ifndef JSON_LOADER_H
#define JSON_LOADER_H

#include "loader.h"

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

		void LoadMaterial(const nlohmann::json& json, Material& material);

		template <World::ContainerType T, typename U = World::type_of_t<T>>
		Handle<U> Load(const nlohmann::json& object_json);

		template <World::ContainerType T, typename U = World::type_of_t<T>>
		void ObjectLoad(const nlohmann::json& world_json, const std::string& key);
		void LoadWorld(const nlohmann::json& world_json);
	};
}

#endif