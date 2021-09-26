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
		Scene m_scene;
		UI::Interface m_ui;
	private:
		bool m_display_info;

	public:
		Application();
		~Application();


	public:
		void Init();
		int Start();
	private:
		void Update();
		void Keyboard_OnKeyPress(WAF::Keyboard::Events::EventKeyPress& event);
	};

}

#endif 