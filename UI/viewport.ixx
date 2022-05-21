module;

#include "rayzath.h"

export module rz.ui.windows.viewport;

export namespace RayZath::UI
{
	class Viewport
	{
	public:
		void draw(Graphics::Bitmap&& bitmap);
	};
}
