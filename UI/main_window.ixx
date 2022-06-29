module;

#include "rayzath.h"

export module rz.ui.windows.main;

import rz.ui.scene;

export namespace RayZath::UI::Windows
{
	class Main
	{
	private:
		Scene& mr_scene;

	public:
		Main(Scene& scene);

		void update();
	};
}
