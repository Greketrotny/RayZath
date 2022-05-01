export module rz.ui.application;

import rz.ui.rendering;

export namespace RayZath::UI
{
	class Application
	{
	private:
		Rendering m_rendering;

	public:
		static Application& instance();
		int run();
	};
}
