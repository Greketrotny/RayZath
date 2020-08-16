#ifndef SETTINGS_WINDOW_H
#define SETTINGS_WINDOW_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

namespace Tester
{
	namespace UI
	{
		class Interface;

		class ControlPanel
		{
		private:
			Interface& mr_iface;


			// ~~~~ interface ~~~~
			WAF::Window* mp_window;

			WAF::Label* mp_lObjectCategory;
			WAF::ComboBox* mp_cbObjectCategory;
			WAF::Label* mp_lObjectList;
			WAF::ComboBox* mp_cbObjectList;

		public:
			ControlPanel(Interface& interf);
			~ControlPanel();
		};
	}
}

#endif