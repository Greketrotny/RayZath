#ifndef SETTINGS_WINDOW_H
#define SETTINGS_WINDOW_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

#include "properties_editors.h"

namespace Tester
{
	namespace UI
	{
		class Interface;

		class ControlPanel
		{
		private:
			Interface& mr_iface;
			RZ::World* mp_world;

			enum class ObjectCategory
			{
				Camera,
				SpotLight,
				DirectLight,
				Mesh,
				Sphere
			} m_curr_object_category;

		public:
			PropsEditor* mp_props_editor;


			// ~~~~ interface ~~~~
			WAF::Window* mp_window;

			WAF::Label* mp_lObjectCategory;
			WAF::ComboBox* mp_cbObjectCategory;
			WAF::Label* mp_lObjectList;
			WAF::ComboBox* mp_cbObjectList;

		public:
			ControlPanel(Interface& interf, RZ::World* world);
			~ControlPanel();


			// ~~~~ event handlers ~~~~
		public:
			void CB_ObjectCategory_OnAccept(WAF::ComboBox::Events::EventSelectionAccept& event);
			void CB_ObjectList_OnAccept(WAF::ComboBox::Events::EventSelectionAccept& event);
		};
	}
}

#endif