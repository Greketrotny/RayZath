#ifndef SAVER_H
#define SAVER_H

#include "world.h"

#include <filesystem>

namespace RayZath::Engine
{
	class Saver
	{
	private:
		World& mr_world;

	public:
		Saver(World& world);

	public:
		void SaveScene(const std::filesystem::path& path);
	};
}

#endif //!LOADER_H
