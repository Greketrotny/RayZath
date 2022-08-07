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
	class SaverBase
	{
	protected:
		World& mr_world;

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
			std::filesystem::path texture, normal, metalness, roughness, emission;
		};

		MTLSaver(World& world)
			: BitmapSaver(world)
		{}

		std::filesystem::path SaveMTL(
			const Handle<Material>& material,
			const std::filesystem::path& path,
			const std::string& file_name,
			const MapsPaths& maps_paths);
		std::filesystem::path SaveMTLWithMaps(
			const Handle<Material>& material,
			const std::filesystem::path& path,
			const std::string& file_name);
	};
	class OBJSaver : public MTLSaver
	{
	public:
		OBJSaver(World& world)
			: MTLSaver(world)
		{}

		void SaveOBJ(
			const std::vector<Handle<MeshStructure>>& meshes,
			const std::filesystem::path& path,
			const std::optional<std::filesystem::path>& material_library,
			const std::unordered_map<uint32_t, std::string>& material_names);
	private:
		void SaveMesh(
			const Handle<MeshStructure>& mesh,
			std::ofstream& file,
			const std::unordered_map<uint32_t, std::string>& material_names);
	};

	class JsonSaver;
	class Saver : public OBJSaver
	{
	private:
		std::unique_ptr<JsonSaver> mp_json_saver;
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
		Saver(World& world);

		void SaveScene(const SaveOptions& options);
	};
}

#endif //!LOADER_H
