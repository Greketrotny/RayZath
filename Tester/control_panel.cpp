#include "control_panel.h"
#include "application.h"

namespace Tester
{
	namespace UI
	{
		ControlPanel::ControlPanel(Interface& interf)
			: mr_iface(interf)
		{
			// ~~~~ interface ~~~~
			// window
			mp_window = WAF::Framework::GetInstance().CreateNewWindow(WAF::ConStruct<WAF::Window>(
				L"Control panel",
				WAF::Rect(WAF::Point(20, 50), WAF::Size(300, 700)),
				WAF::Size(300, 100), WAF::Size(330, 1000)));
			mp_window->BindEventFunc(&Interface::ControlPanel_OnClose, &mr_iface);

			// object category choice
			mp_lObjectCategory = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 10, 140, 50), L"Object category: ", WAF::Label::TextAlignment::Right));
			mp_cbObjectCategory = mp_window->CreateChild(WAF::ConStruct<WAF::ComboBox>(
				WAF::Rect(150, 10, 140, 50)));
			mp_cbObjectCategory->AddItem(L"Camera");
			mp_cbObjectCategory->AddItem(L"Point light");
			mp_cbObjectCategory->AddItem(L"Spot light");
			mp_cbObjectCategory->AddItem(L"Direct light");
			mp_cbObjectCategory->AddItem(L"Mesh");
			mp_cbObjectCategory->AddItem(L"Sphere");

			// object list
			mp_lObjectList = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 40, 140, 50), L"Object: ", WAF::Label::TextAlignment::Right));
			mp_cbObjectList = mp_window->CreateChild(WAF::ConStruct<WAF::ComboBox>(
				WAF::Rect(150, 40, 140, 50)));
		}
		ControlPanel::~ControlPanel()
		{
			mp_window->Destroy();
		}
	}
}