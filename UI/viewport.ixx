module;

#include "rayzath.h"
#include "imgui.h"

export module rz.ui.windows.viewport;

export namespace RayZath::UI
{
	class Viewport
	{
	public:
		void setNewBitmap(Graphics::Bitmap&& bitmap);
		void draw(const ImTextureID image, const float width, const float height);
	};
}
