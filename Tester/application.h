#ifndef APPLICATION_H
#define APPLICATION_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

#include "interface.h"
#include "scene.h"

namespace Tester
{
	class Application
	{
	public:
		UI::Interface m_ui;
		Scene m_scene;

	public:
		Application();
		~Application();


	public:
		int Start();
	private:
		void Update();
	};

}

#endif 