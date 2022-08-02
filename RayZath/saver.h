#ifndef SAVER_H
#define SAVER_H

#include "world.h"

#include <filesystem>

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


		void SaveAllMaps(const std::filesystem::path& path);
		template <World::ObjectType T>
		void SaveAllTypeMaps(const std::filesystem::path& path)
		{
			auto& maps = mr_world.Container<T>();
			if (maps.GetCount() == 0) return;
			std::filesystem::create_directory(path);
			for (uint32_t id = 0; id < maps.GetCount(); id++)
			{
				auto& map = maps[id];
				if (!map) continue;
				SaveMap<T>(map->GetBitmap(), path, map->GetName());
			}
		}
		template <World::ObjectType T>
		void SaveMap(
			const typename World::object_t<T>::buffer_t& map,
			const std::filesystem::path& path,
			const std::string& file_name);
	};
	class Saver : public BitmapSaver
	{
	public:
		struct SaveOptions
		{
			std::filesystem::path path{};
			bool allow_partial_write = true;
		};

	public:
		Saver(World& world);

	public:
		void SaveScene(const SaveOptions& options);
	};
}

#endif //!LOADER_H
