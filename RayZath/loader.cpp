#include "loader.h"

#include "./lib/include/CImg.h"

namespace RayZath
{
	// ~~~~~~~~ BitmapLoader ~~~~~~~~
	BitmapLoader::BitmapLoader(World& world)
		: LoaderBase(world)
	{}

	Graphics::Bitmap BitmapLoader::LoadTexture(const std::string& path)
	{
		cimg_library::cimg::imagemagick_path(
			"D:/Program Files/ImageMagick-7.0.10-53-portable-Q8-x64/convert.exe");
		cil::CImg<unsigned char> image(path.c_str());

		Graphics::Bitmap texture(image.width(), image.height());
		if (image.spectrum() == 3)
		{
			for (int x = 0; x < texture.GetWidth(); x++)
			{
				for (int y = 0; y < texture.GetHeight(); y++)
				{
					texture.Value(x, y) =
						Graphics::Color(
							*image.data(x, y, 0, 0),
							*image.data(x, y, 0, 1),
							*image.data(x, y, 0, 2), 0xFF);
				}
			}
		}
		else if (image.spectrum() == 1)
		{
			for (int x = 0; x < texture.GetWidth(); x++)
			{
				for (int y = 0; y < texture.GetHeight(); y++)
				{
					auto& value = *image.data(x, y, 0, 0);
					texture.Value(x, y) =
						Graphics::Color(value);
				}
			}
		}

		return texture;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ MTLLoader ~~~~~~~~
	MTLLoader::MTLLoader(World& world)
		: BitmapLoader(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ OBJLoader ~~~~~~~~
	OBJLoader::OBJLoader(World& world)
		: MTLLoader(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~



	// ~~~~~~~~ Loader ~~~~~~~~
	Loader::Loader(World& world)
		: OBJLoader(world)
	{}
	// ~~~~~~~~~~~~~~~~~~~~~~~~
}