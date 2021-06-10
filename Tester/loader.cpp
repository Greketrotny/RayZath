#include "loader.h"
#include "CImg.h"

namespace Tester
{
	Graphics::Bitmap LoadFromFile(const char* file)
	{
		cimg_library::cimg::imagemagick_path(
			"D:/Program Files/ImageMagick-7.0.10-53-portable-Q8-x64/convert.exe");
		cil::CImg<unsigned char> image(file);

		Graphics::Bitmap bitmap(image.width(), image.height());
		if (image.spectrum() == 3)
		{
			for (int x = 0; x < bitmap.GetWidth(); x++)
			{
				for (int y = 0; y < bitmap.GetHeight(); y++)
				{
					bitmap.Value(x, y) =
						Graphics::Color(
							*image.data(x, y, 0, 0),
							*image.data(x, y, 0, 1),
							*image.data(x, y, 0, 2), 0xFF);
				}
			}
		}
		else if (image.spectrum() == 1)
		{
			for (int x = 0; x < bitmap.GetWidth(); x++)
			{
				for (int y = 0; y < bitmap.GetHeight(); y++)
				{
					auto& value = *image.data(x, y, 0, 0);
					bitmap.Value(x, y) =
						Graphics::Color(value);
				}
			}
		}

		

		return bitmap;
	}
}