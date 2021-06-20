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
			, m_curr_object_category(ObjectCategory::Camera)
		{
			// ~~~~ interface ~~~~
			// window
			mp_window = WAF::Framework::GetInstance().CreateNewWindow(WAF::ConStruct<WAF::Window>(
				L"Control panel",
				WAF::Rect(WAF::Point(5, 40), WAF::Size(310, 700)),
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

		std::wstring ToWstring(const std::string& s)
		{
			return { s.begin(), s.end() };
		}
		void ControlPanel::CB_ObjectCategory_OnAccept(WAF::ComboBox::Events::EventSelectionAccept& event)
		{
			const std::wstring sel_name = mp_cbObjectCategory->GetSelectedItem();

			if (sel_name == L"Camera")
			{
				m_curr_object_category = ObjectCategory::Camera;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::Camera>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::Camera>()[i]->GetName()));
			}
			else if (sel_name == L"Point light")
			{
				m_curr_object_category = ObjectCategory::PointLight;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::PointLight>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::PointLight>()[i]->GetName()));
			}
			else if (sel_name == L"Spot light")
			{
				m_curr_object_category = ObjectCategory::SpotLight;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::SpotLight>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::SpotLight>()[i]->GetName()));
			}
			else if (sel_name == L"Direct light")
			{
				m_curr_object_category = ObjectCategory::DirectLight;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::DirectLight>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::DirectLight>()[i]->GetName()));
			}
			else if (sel_name == L"Mesh")
			{
				m_curr_object_category = ObjectCategory::Mesh;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::Mesh>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::Mesh>()[i]->GetName()));
			}
			else if (sel_name == L"Sphere")
			{
				m_curr_object_category = ObjectCategory::Sphere;
				mp_cbObjectList->Clear();

				for (int i = 0; i < mp_world->Container<RZ::World::ContainerType::Sphere>().GetCount(); i++)
					mp_cbObjectList->AddItem(ToWstring(mp_world->Container<RZ::World::ContainerType::Sphere>()[i]->GetName()));
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
						mp_world->Container<RZ::World::ContainerType::Camera>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::PointLight:
					mp_props_editor = new PointLightEditor(
						mp_window, 
						mp_world->Container<RZ::World::ContainerType::PointLight>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::SpotLight:
					mp_props_editor = new SpotLightEditor(
						mp_window,
						mp_world->Container<RZ::World::ContainerType::SpotLight>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::DirectLight:
					mp_props_editor = new DirectLightEditor(
						mp_window,
						mp_world->Container<RZ::World::ContainerType::DirectLight>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::Sphere:
					mp_props_editor = new SphereEditor(
						mp_window, mp_world->Container<RZ::World::ContainerType::Sphere>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
				case Tester::UI::ControlPanel::ObjectCategory::Mesh:
					mp_props_editor = new MeshEditor(
						mp_window, 
						mp_world->Container<RZ::World::ContainerType::Mesh>()[mp_cbObjectList->GetSelectedItemIndex()]);
					break;
			}
		}
	}
}