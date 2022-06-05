export module rz.ui.windows.explorer;

import rz.ui.scene;

namespace RayZath::UI::Windows
{
	export class Explorer
	{
	private:
		Scene& mr_scene;

	public:
		Explorer(Scene& scene);

		void update();

	private:
		void listLights();
	};
}