#ifndef LOADER_H
#define LOADER_H

#include "world.h"

#include <string>

namespace RayZath
{
	class LoaderBase
	{
	private:
		World& mr_world;


	public:
		LoaderBase(World& world)
			: mr_world(world)
		{}
	};

	class BitmapLoader
		: public LoaderBase
	{
	public:
		BitmapLoader(World& world);

	public:
		static Graphics::Bitmap LoadTexture(const std::string& path);
	};

	class MTLLoader
		: public BitmapLoader
	{
	public:
		MTLLoader(World& world);


	public:
		static std::vector<Handle<Material>> LoadMTL(const std::string& path);
	};

	class OBJLoader
		: public MTLLoader
	{
	public:
		OBJLoader(World& world);
	};

	class Loader
		: public OBJLoader
	{
	public:
		Loader(World& world);


	public:
		// TODO:
		// static void LoadScene(const std::string& path);
	};
}

#endif //!LOADER_H