export module rz.ui.windows.explorer;

import rz.ui.scene;
import rz.ui.windows.properties;

import <memory>;

namespace RayZath::UI::Windows
{
	export class Explorer
	{
	private:
		Scene& mr_scene;
		Properties m_properties;

	public:
		Explorer(Scene& scene);

		void update();

	private:
		void listLights();
	};
}