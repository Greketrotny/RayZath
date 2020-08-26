#include "control_panel.h"
#include "application.h"

namespace Tester
{
	namespace UI
	{
		ControlPanel::ControlPanel(Interface& interf, RZ::World* world)
			: mr_iface(interf)
			, mp_world(world)
			, mp_props_editor(nullptr)
		{
			// ~~~~ interface ~~~~
			// window
			mp_window = WAF::Framework::GetInstance().CreateNewWindow(WAF::ConStruct<WAF::Window>(
				L"Control panel",
				WAF::Rect(WAF::Point(20, 50), WAF::Size(310, 700)),
				WAF::Size(310, 100), WAF::Size(330, 1000)));
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
			mp_cbObjectCategory->BindEventFunc(&ControlPanel::CB_ObjectCategory_OnAccept, this);

			// object list
			mp_lObjectList = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 40, 140, 50), L"Object: ", WAF::Label::TextAlignment::Right));
			mp_cbObjectList = mp_window->CreateChild(WAF::ConStruct<WAF::ComboBox>(
				WAF::Rect(150, 40, 140, 50)));
			mp_cbObjectList->BindEventFunc(&ControlPanel::CB_ObjectList_OnAccept, this);
		}
		ControlPanel::~ControlPanel()
		{
			if (mp_props_editor) delete mp_props_editor;

			mp_window->Destroy();
		}

		void ControlPanel::CB_ObjectCategory_OnAccept(WAF::ComboBox::Events::EventSelectionAccept& event)
		{
			const std::wstring sel_name = mp_cbObjectCategory->GetSelectedItem();

			if (sel_name == L"Camera")
			{
				m_curr_object_category = ObjectCategory::Camera;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->GetCameras().GetCount(); i++)
					mp_cbObjectList->AddItem(mp_world->GetCameras()[i]->GetName());
			}
			else if (sel_name == L"Point light")
			{
				m_curr_object_category = ObjectCategory::PointLight;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->GetPointLights().GetCount(); i++)
					mp_cbObjectList->AddItem(mp_world->GetPointLights()[i]->GetName());
			}
			else if (sel_name == L"Spot light")
			{
				m_curr_object_category = ObjectCategory::SpotLight;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->GetSpotLights().GetCount(); i++)
					mp_cbObjectList->AddItem(mp_world->GetSpotLights()[i]->GetName());
			}
			else if (sel_name == L"Mesh")
			{
				m_curr_object_category = ObjectCategory::Mesh;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->GetMeshes().GetCount(); i++)
					mp_cbObjectList->AddItem(mp_world->GetMeshes()[i]->GetName());
			}
			else if (sel_name == L"Sphere")
			{
				m_curr_object_category = ObjectCategory::Sphere;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->GetSpheres().GetCount(); i++)
					mp_cbObjectList->AddItem(mp_world->GetSpheres()[i]->GetName());
			}
		}
		void ControlPanel::CB_ObjectList_OnAccept(WAF::ComboBox::Events::EventSelectionAccept& event)
		{
			if (mp_props_editor) delete mp_props_editor;

			switch (m_curr_object_category)
			{
				case Tester::UI::ControlPanel::ObjectCategory::Camera:
					mp_props_editor = new CameraPropsEditor(
						mp_window, 
						mp_world->GetCameras()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::PointLight:
					mp_props_editor = new PointLightEditor(
						mp_window, 
						mp_world->GetPointLights()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::SpotLight:
					mp_props_editor = new SpotLightEditor(
						mp_window,
						mp_world->GetSpotLights()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::DirectLight:
					break;
				case Tester::UI::ControlPanel::ObjectCategory::Mesh:
					mp_props_editor = new MeshEditor(
						mp_window, 
						mp_world->GetMeshes()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::Sphere:
					mp_props_editor = new SphereEditor(
						mp_window, mp_world->GetSpheres()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
			}
		}
	}
}