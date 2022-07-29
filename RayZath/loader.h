#ifndef LOADER_H
#define LOADER_H

#include "world.h"

#include <string>
#include <array>
#include <filesystem>
#include <memory>

namespace RayZath::Engine
{
	class JsonLoader;

	class LoaderBase
	{
	protected:
		World& mr_world;

	public:
		LoaderBase(World& world);
	};

	class BitmapLoader
		: public LoaderBase
	{
	public:
		BitmapLoader(World& world);

	public:
		Graphics::Bitmap LoadTexture(const std::string& path);
		Graphics::Bitmap LoadNormalMap(const std::string& path);
		Graphics::Buffer2D<uint8_t> LoadMetalnessMap(const std::string& path);
		Graphics::Buffer2D<uint8_t> LoadRoughnessMap(const std::string& path);
		Graphics::Buffer2D<float> LoadEmissionMap(const std::string& path);
	};

	class MTLLoader
		: public BitmapLoader
	{
	public:
		MTLLoader(World& world);

	public:
		std::vector<Handle<Material>> LoadMTL(const std::filesystem::path& path);
		void LoadMTL(const std::filesystem::path& path, Material& material);
	private:
	};

	class OBJLoader
		: public MTLLoader
	{
	public:
		OBJLoader(World& world);

	public:
		Handle<Group> LoadOBJ(const std::filesystem::path& path);
	};

	class Loader
		: public OBJLoader
	{
	private:
		std::unique_ptr<JsonLoader> mp_json_loader;

	public:
		Loader(World& world);

	public:
		 void LoadScene(const std::filesystem::path& path);
	};
}

#endif //!LOADER_H
