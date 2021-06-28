#ifndef LOADER_H
#define LOADER_H

#include "world.h"

#include <string>
#include <array>

namespace RayZath
{
	class LoaderBase
	{
	protected:
		World& mr_world;


	public:
		LoaderBase(World& world);


	protected:
		std::array<std::string, 3ull> ParseFileName(const std::string & file_name);
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
		Graphics::Buffer2D<uint8_t> LoadSpecularityMap(const std::string& path);
		Graphics::Buffer2D<uint8_t> LoadRoughnessMap(const std::string& path);
	};

	class MTLLoader
		: public BitmapLoader
	{
	public:
		MTLLoader(World& world);


	public:
		std::vector<Handle<Material>> LoadMTL(const std::string& path);
	private:
	};

	class OBJLoader
		: public MTLLoader
	{
	public:
		OBJLoader(World& world);


	public:
		std::vector<Handle<Mesh>> LoadOBJ(const std::string& path);
	};

	class Loader
		: public OBJLoader
	{
	public:
		Loader(World& world);


	public:
		// TODO:
		// void LoadScene(const std::string& path);
	};
}

#endif //!LOADER_H