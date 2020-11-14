#include "properties_editors.h"

#include <string>

namespace Tester
{
	namespace UI
	{
		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		PositionEditor::PositionEditor(WAF::Window* window, RZ::RenderObject* object,
			const WAF::Point& position)
			: mp_window(window)
			, mp_object(object)
		{
			mp_pPosition = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 150)));
			mp_lPosition = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"position:", WAF::Label::TextAlignment::Center));

			mp_lPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-1000, 1000),
				mp_object->GetPosition().x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbPosX->BindEventFunc(&PositionEditor::TBPositionX_OnDrag, this);

			mp_lPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-1000, 1000),
				mp_object->GetPosition().y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbPosY->BindEventFunc(&PositionEditor::TBPositionY_OnDrag, this);

			mp_lPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-1000, 1000),
				mp_object->GetPosition().z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbPosZ->BindEventFunc(&PositionEditor::TBPositionZ_OnDrag, this);

			WritePosition(mp_object->GetPosition());
		}
		PositionEditor::~PositionEditor()
		{
			mp_pPosition->Destroy();
		}

		void PositionEditor::WritePosition(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lPosX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lPosY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lPosZ->SetCaption(buffer);
		}

		void PositionEditor::TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetPosition(Math::vec3<float>(
				mp_tbPosX->GetPosition() / 100.0f,
				mp_object->GetPosition().y,
				mp_object->GetPosition().z));
			WritePosition(mp_object->GetPosition());
		}
		void PositionEditor::TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetPosition(Math::vec3<float>(
				mp_object->GetPosition().x,
				mp_tbPosY->GetPosition() / 100.0f,
				mp_object->GetPosition().z));
			WritePosition(mp_object->GetPosition());
		}
		void PositionEditor::TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetPosition(Math::vec3<float>(
				mp_object->GetPosition().x,
				mp_object->GetPosition().y,
				mp_tbPosZ->GetPosition() / 100.0f));
			WritePosition(mp_object->GetPosition());
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		RotationEditor::RotationEditor(WAF::Window* window, RZ::RenderObject* object,
			const WAF::Point& position)
			: mp_window(window)
			, mp_object(object)
		{
			mp_pRotation = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 150)));
			mp_lRotation = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"rotation:", WAF::Label::TextAlignment::Center));

			mp_lRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-314, 314),
				mp_object->GetRotation().x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbRotX->BindEventFunc(&RotationEditor::TBRotationX_OnDrag, this);

			mp_lRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-314, 314),
				mp_object->GetRotation().y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbRotY->BindEventFunc(&RotationEditor::TBRotationY_OnDrag, this);

			mp_lRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-314, 314),
				mp_object->GetRotation().z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbRotZ->BindEventFunc(&RotationEditor::TBRotationZ_OnDrag, this);

			WriteRotation(mp_object->GetRotation());
		}
		RotationEditor::~RotationEditor()
		{
			mp_pRotation->Destroy();
		}

		void RotationEditor::WriteRotation(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lRotX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lRotY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lRotZ->SetCaption(buffer);
		}

		void RotationEditor::TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetRotation(Math::vec3<float>(
				mp_tbRotX->GetPosition() / 100.0f,
				mp_object->GetRotation().y,
				mp_object->GetRotation().z));
			WriteRotation(mp_object->GetRotation());
		}
		void RotationEditor::TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetRotation(Math::vec3<float>(
				mp_object->GetRotation().x,
				mp_tbRotY->GetPosition() / 100.0f,
				mp_object->GetRotation().z));
			WriteRotation(mp_object->GetRotation());
		}
		void RotationEditor::TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetRotation(Math::vec3<float>(
				mp_object->GetRotation().x,
				mp_object->GetRotation().y,
				mp_tbRotZ->GetPosition() / 100.0f));
			WriteRotation(mp_object->GetRotation());
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		ScaleEditor::ScaleEditor(WAF::Window* window, RZ::RenderObject* object,
			const WAF::Point& position)
			: mp_window(window)
			, mp_object(object)
		{
			mp_pScale = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 150)));
			mp_lScale = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"scale:", WAF::Label::TextAlignment::Center));

			mp_lScaleX = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbScaleX = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(1, 500),
				mp_object->GetScale().x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbScaleX->BindEventFunc(&ScaleEditor::TBScaleX_OnDrag, this);

			mp_lScaleY = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbScaleY = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(1, 500),
				mp_object->GetScale().y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbScaleY->BindEventFunc(&ScaleEditor::TBScaleY_OnDrag, this);

			mp_lScaleZ = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbScaleZ = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(1, 500),
				mp_object->GetScale().z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				100u, false));
			mp_tbScaleZ->BindEventFunc(&ScaleEditor::TBScaleZ_OnDrag, this);

			WriteScale(mp_object->GetScale());
		}
		ScaleEditor::~ScaleEditor()
		{
			mp_pScale->Destroy();
		}

		void ScaleEditor::WriteScale(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lScaleX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lScaleY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lScaleZ->SetCaption(buffer);
		}

		void ScaleEditor::TBScaleX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetScale(Math::vec3<float>(
				mp_tbScaleX->GetPosition() / 100.0f,
				mp_object->GetScale().y,
				mp_object->GetScale().z));
			WriteScale(mp_object->GetScale());
		}
		void ScaleEditor::TBScaleY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetScale(Math::vec3<float>(
				mp_object->GetScale().x,
				mp_tbScaleY->GetPosition() / 100.0f,
				mp_object->GetScale().z));
			WriteScale(mp_object->GetScale());
		}
		void ScaleEditor::TBScaleZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->SetScale(Math::vec3<float>(
				mp_object->GetScale().x,
				mp_object->GetScale().y,
				mp_tbScaleZ->GetPosition() / 100.0f));
			WriteScale(mp_object->GetScale());
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] MaterialEditor ~~~~~~~~
		MaterialEditor::MaterialEditor(
			WAF::Window* window, 
			RZ::RenderObject* object,
			const WAF::Point& position)
			: mp_window(window)
			, mp_object(object)
		{
			// panel and header
			mp_pMaterial = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 250)));
			mp_lMaterial = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"material:", WAF::Label::TextAlignment::Center));

			// reflectance
			mp_lReflectance = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 25, 100, 15), L"Reflectance:"));
			mp_tbReflectance = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 40, 245, 25),
				WAF::Range(0, 100),
				mp_object->GetMaterial().GetReflectance() * 100.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbReflectance->BindEventFunc(&MaterialEditor::TBReflectance_OnDrag, this);

			// glossiness
			mp_lGlossiness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 75, 100, 15), L"Glossiness:"));
			mp_tbGlossiness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 90, 245, 25),
				WAF::Range(0, 100),
				mp_object->GetMaterial().GetGlossiness() * 10000.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbGlossiness->BindEventFunc(&MaterialEditor::TBGlossiness_OnDrag, this);

			// transmittance
			mp_lTransmittance = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 125, 100, 15), L"Transmittance:"));
			mp_tbTransmittance = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 140, 245, 25),
				WAF::Range(0, 100),
				mp_object->GetMaterial().GetTransmittance() * 100.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbTransmittance->BindEventFunc(&MaterialEditor::TBTransmittance_OnDrag, this);

			// IOR
			mp_lIOR = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 175, 150, 15), L"Refraction index:"));
			mp_tbIOR = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 190, 245, 25),
				WAF::Range(100, 500),
				mp_object->GetMaterial().GetIndexOfRefraction() * 100.0f,
				1u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				100u, false));
			mp_tbIOR->BindEventFunc(&MaterialEditor::TBIOR_OnDrag, this);

			// Emittance
			mp_lEmission = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 225, 50, 15), L"Emission:"));
			mp_eEmission = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(60, 223, 100, 20), std::to_wstring(int(mp_object->GetMaterial().GetEmittance())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&MaterialEditor::EEmission_OnEdit, this);

			WriteMaterialProps();
		}
		MaterialEditor::~MaterialEditor()
		{
			mp_pMaterial->Destroy();
		}

		void MaterialEditor::WriteMaterialProps()
		{
			const size_t buff_size = 32;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"Reflectance: %1.2f", mp_object->GetMaterial().GetReflectance());
			mp_lReflectance->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Glossiness: %1.4f", mp_object->GetMaterial().GetGlossiness());
			mp_lGlossiness->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Transmittance: %1.2f", mp_object->GetMaterial().GetTransmittance());
			mp_lTransmittance->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Refraction index: %1.2f", mp_object->GetMaterial().GetIndexOfRefraction());
			mp_lIOR->SetCaption(buffer);
		}

		void MaterialEditor::TBReflectance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->GetMaterial().SetReflectance(
				mp_tbReflectance->GetPosition() / 
				static_cast<float>(mp_tbReflectance->GetMaxTrackValue()));
			mp_object->GetStateRegister().RequestUpdate();
			WriteMaterialProps();
		}
		void MaterialEditor::TBGlossiness_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->GetMaterial().SetGlossiness(
				mp_tbGlossiness->GetPosition() / 10000.0f);
			mp_object->GetStateRegister().RequestUpdate();
			WriteMaterialProps();
		}
		void MaterialEditor::TBTransmittance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->GetMaterial().SetTransmittance(
				mp_tbTransmittance->GetPosition() /
				static_cast<float>(mp_tbTransmittance->GetMaxTrackValue()));
			mp_object->GetStateRegister().RequestUpdate();
			WriteMaterialProps();
		}
		void MaterialEditor::TBIOR_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_object->GetMaterial().SetIndexOfRefraction(
				mp_tbIOR->GetPosition() / 100.0f);
			mp_object->GetStateRegister().RequestUpdate();
			WriteMaterialProps();
		}
		void MaterialEditor::EEmission_OnEdit(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				mp_object->GetMaterial().SetEmittance(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				mp_object->GetMaterial().SetEmittance(0.0f);
			}
			mp_object->GetStateRegister().RequestUpdate();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



		// ~~~~~~~~ [CLASS] CameraPropsEditor ~~~~~~~~
		CameraPropsEditor::CameraPropsEditor(WAF::Window* window, RZ::Camera* camera)
			: mp_window(window)
			, mp_camera(camera)
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 500), L"Camera properties"));

			// position
			mp_pPosition = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 100, 260, 150)));
			mp_lPosition = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"position:", WAF::Label::TextAlignment::Center));

			mp_lPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-100, 100),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosX->BindEventFunc(&CameraPropsEditor::TBPositionX_OnDrag, this);

			mp_lPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-100, 100),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosY->BindEventFunc(&CameraPropsEditor::TBPositionY_OnDrag, this);

			mp_lPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-100, 100),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosZ->BindEventFunc(&CameraPropsEditor::TBPositionZ_OnDrag, this);


			// rotation
			mp_pRotation = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 260, 260, 150)));
			mp_lRotation = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"rotation:", WAF::Label::TextAlignment::Center));

			mp_lRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbRotX->BindEventFunc(&CameraPropsEditor::TBRotationX_OnDrag, this);

			mp_lRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbRotY->BindEventFunc(&CameraPropsEditor::TBRotationY_OnDrag, this);

			mp_lRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbRotZ->BindEventFunc(&CameraPropsEditor::TBRotationZ_OnDrag, this);

			// others
			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 420, 260, 150)));

			// fov
			mp_lFov = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 50, 20), L"Fov: "));
			mp_tbFov = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 0, 200, 40),
				WAF::Range(10, 300),
				mp_camera->GetFov().value() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbFov->BindEventFunc(&CameraPropsEditor::TBFov_OnDrag, this);

			// focal distance
			mp_lFocalDistance = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 50, 50, 30), L"Focal distance: "));
			mp_tbFocalDistance = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 40, 200, 40),
				WAF::Range(10, 3000),
				mp_camera->GetFocalDistance() * 100.0f, 10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				200u, false));
			mp_tbFocalDistance->BindEventFunc(&CameraPropsEditor::TBFocalDistance_OnDrag, this);

			// aperature
			mp_lAperature = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 95, 50, 30), L"Aperature: "));
			mp_tbAperature = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 80, 200, 40),
				WAF::Range(0, 100),
				mp_camera->GetAperture() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbAperature->BindEventFunc(&CameraPropsEditor::TBAperature_OnDrag, this);
		}
		CameraPropsEditor::~CameraPropsEditor()
		{
			mp_gbProperties->Destroy();

			mp_pPosition->Destroy();
			mp_pRotation->Destroy();
			mp_pOthers->Destroy();
		}

		void CameraPropsEditor::UpdateState()
		{
			mp_tbPosX->SetThumbPosition(mp_camera->GetPosition().x * 10.0f);
			mp_tbPosY->SetThumbPosition(mp_camera->GetPosition().y * 10.0f);
			mp_tbPosZ->SetThumbPosition(mp_camera->GetPosition().z * 10.0f);
			WritePosition(mp_camera->GetPosition());

			mp_tbRotX->SetThumbPosition(mp_camera->GetRotation().x * 100.0f);
			mp_tbRotY->SetThumbPosition(mp_camera->GetRotation().y * 100.0f);
			mp_tbRotZ->SetThumbPosition(mp_camera->GetRotation().z * 100.0f);
			WriteRotation(mp_camera->GetRotation());
		}

		void CameraPropsEditor::WritePosition(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lPosX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lPosY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lPosZ->SetCaption(buffer);
		}
		void CameraPropsEditor::WriteRotation(const Math::vec3<float>& rot)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", rot.x);
			mp_lRotX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", rot.y);
			mp_lRotY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", rot.z);
			mp_lRotZ->SetCaption(buffer);
		}

		// event handlers 
		void CameraPropsEditor::TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetPosition(Math::vec3<float>(
				mp_tbPosX->GetPosition() / 10.0f,
				mp_camera->GetPosition().y,
				mp_camera->GetPosition().z));
			WritePosition(mp_camera->GetPosition());
		}
		void CameraPropsEditor::TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetPosition(Math::vec3<float>(
				mp_camera->GetPosition().x,
				mp_tbPosY->GetPosition() / 10.0f,
				mp_camera->GetPosition().z));
			WritePosition(mp_camera->GetPosition());
		}
		void CameraPropsEditor::TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetPosition(Math::vec3<float>(
				mp_camera->GetPosition().x,
				mp_camera->GetPosition().y,
				mp_tbPosZ->GetPosition() / 10.0f));
			WritePosition(mp_camera->GetPosition());
		}

		void CameraPropsEditor::TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			float rotX = mp_tbRotX->GetPosition() / 100.0f;
			mp_camera->SetRotation(Math::vec3<float>(
				rotX,
				mp_camera->GetRotation().y,
				mp_camera->GetRotation().z));
			WriteRotation(mp_camera->GetRotation());
		}
		void CameraPropsEditor::TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetRotation(Math::vec3<float>(
				mp_camera->GetRotation().x,
				mp_tbRotY->GetPosition() / 100.0f,
				mp_camera->GetRotation().z));
			WriteRotation(mp_camera->GetRotation());
		}
		void CameraPropsEditor::TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetRotation(Math::vec3<float>(
				mp_camera->GetRotation().x,
				mp_camera->GetRotation().y,
				mp_tbRotZ->GetPosition() / 100.0f));
			WriteRotation(mp_camera->GetRotation());
		}

		void CameraPropsEditor::TBFov_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetFov(mp_tbFov->GetPosition() / 100.0f);
		}
		void CameraPropsEditor::TBFocalDistance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetFocalDistance(mp_tbFocalDistance->GetPosition() / 100.0f);
		}
		void CameraPropsEditor::TBAperature_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_camera->SetAperture(mp_tbAperature->GetPosition() / 100.0f);
		}



		// ~~~~~~~~ [CLASS] PointLightEditor ~~~~~~~~
		PointLightEditor::PointLightEditor(WAF::Window* window, RZ::PointLight* light)
			: mp_window(window)
			, mp_light(light)
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 500), L"Point light properties"));

			// position
			mp_pPosition = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 100, 260, 150)));
			mp_lPosition = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"position:", WAF::Label::TextAlignment::Center));

			mp_lPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().x * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosX->BindEventFunc(&PointLightEditor::TBPositionX_OnDrag, this);

			mp_lPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().y * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosY->BindEventFunc(&PointLightEditor::TBPositionY_OnDrag, this);

			mp_lPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().z * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosZ->BindEventFunc(&PointLightEditor::TBPositionZ_OnDrag, this);
			WritePosition(mp_light->GetPosition());

			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 260, 260, 150)));

			// size
			mp_lSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 40, 15),
				L"Size: "));
			mp_tbSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 0, 200, 40),
				WAF::Range(1, 100),
				mp_light->GetSize() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				50u, false));
			mp_tbSize->BindEventFunc(&PointLightEditor::TBSize_OnDrag, this);

			// emission
			mp_lEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 45, 40, 15),
				L"Emission: "));
			mp_eEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(60, 43, 100, 20), std::to_wstring(int(mp_light->GetEmission())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&PointLightEditor::EditEmission_OnInput, this);
		}
		PointLightEditor::~PointLightEditor()
		{
			mp_gbProperties->Destroy();

			mp_pPosition->Destroy();

			mp_pOthers->Destroy();
		}

		void PointLightEditor::WritePosition(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lPosX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lPosY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lPosZ->SetCaption(buffer);
		}

		void PointLightEditor::TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_tbPosX->GetPosition() / 10.0f,
				mp_light->GetPosition().y,
				mp_light->GetPosition().z));
			WritePosition(mp_light->GetPosition());
		}
		void PointLightEditor::TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_light->GetPosition().x,
				mp_tbPosY->GetPosition() / 10.0f,
				mp_light->GetPosition().z));
			WritePosition(mp_light->GetPosition());
		}
		void PointLightEditor::TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_light->GetPosition().x,
				mp_light->GetPosition().y,
				mp_tbPosZ->GetPosition() / 10.0f));
			WritePosition(mp_light->GetPosition());
		}

		void PointLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetSize(mp_tbSize->GetPosition() / 100.0f);
		}
		void PointLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				mp_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				mp_light->SetEmission(1.0f);
			}
		}

		// ~~~~~~~~ [CLASS] SpotLightEditor ~~~~~~~~
		SpotLightEditor::SpotLightEditor(WAF::Window* window, RZ::SpotLight* light)
			: mp_window(window)
			, mp_light(light)
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 500), L"Spot light properties"));

			// position
			mp_pPosition = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 100, 260, 150)));
			mp_lPosition = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"position:", WAF::Label::TextAlignment::Center));

			mp_lPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"X:"));
			mp_tbPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 20, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().x * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosX->BindEventFunc(&SpotLightEditor::TBPositionX_OnDrag, this);

			mp_lPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 30), L"Y:"));
			mp_tbPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 60, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().y * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosY->BindEventFunc(&SpotLightEditor::TBPositionY_OnDrag, this);

			mp_lPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 30), L"Z:"));
			mp_tbPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 100, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetPosition().z * 10.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbPosZ->BindEventFunc(&SpotLightEditor::TBPositionZ_OnDrag, this);
			WritePosition(mp_light->GetPosition());

			// direction
			mp_pDirection = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 260, 260, 110)));
			mp_lDirection = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"direction:", WAF::Label::TextAlignment::Center));

			mp_lPhi = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 30), L"Phi:"));
			mp_tbPhi = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 20, 180, 40),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				50u, false));
			mp_tbPhi->BindEventFunc(&SpotLightEditor::TBPhi_OnDrag, this);

			mp_lTheta = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 70, 30), L"Theta:"));
			mp_tbTheta = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 60, 180, 40),
				WAF::Range(-157, 157),
				0, 5u, 20u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbTheta->BindEventFunc(&SpotLightEditor::TBTheta_OnDrag, this);
			WriteDirection();


			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 380, 260, 150)));

			// size
			mp_lSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 60, 15),
				L"Size: "));
			mp_tbSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 0, 180, 40),
				WAF::Range(1, 100),
				mp_light->GetSize() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbSize->BindEventFunc(&SpotLightEditor::TBSize_OnDrag, this);
			WriteSize();

			// emission
			mp_lEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 45, 40, 15),
				L"Emission: "));
			mp_eEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(60, 43, 100, 20), std::to_wstring(int(mp_light->GetEmission())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&SpotLightEditor::EditEmission_OnInput, this);

			// beam angle
			mp_lAngle = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 60, 25),
				L"Beam\nangle: "));
			mp_tbAngle = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 70, 180, 40),
				WAF::Range(1, 314),
				mp_light->GetBeamAngle() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				50u, false));
			mp_tbAngle->BindEventFunc(&SpotLightEditor::TBAngle_OnDrag, this);
			WriteAngle();
		}
		SpotLightEditor::~SpotLightEditor()
		{
			mp_gbProperties->Destroy();

			mp_pPosition->Destroy();
			mp_pDirection->Destroy();

			mp_pOthers->Destroy();
		}

		void SpotLightEditor::UpdateState()
		{
			phi = atan2f(mp_light->GetDirection().z, mp_light->GetDirection().x);
			mp_tbPhi->SetThumbPosition(phi * 100.0f);

			theta = asinf(mp_light->GetDirection().y);
			mp_tbTheta->SetThumbPosition(theta * 100.0f);

			WriteDirection();
		}

		void SpotLightEditor::WritePosition(const Math::vec3<float>& pos)
		{
			wchar_t buffer[10];

			std::swprintf(buffer, 10, L"X: %1.2f", pos.x);
			mp_lPosX->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Y: %1.2f", pos.y);
			mp_lPosY->SetCaption(buffer);
			std::swprintf(buffer, 10, L"Z: %1.2f", pos.z);
			mp_lPosZ->SetCaption(buffer);
		}
		void SpotLightEditor::WriteDirection()
		{
			wchar_t buffer[32];

			std::swprintf(buffer, 32, L"Phi: %1.2f", phi);
			mp_lPhi->SetCaption(buffer);
			std::swprintf(buffer, 32, L"Theta: %1.2f", theta);
			mp_lTheta->SetCaption(buffer);
		}
		void SpotLightEditor::WriteSize()
		{
			wchar_t buffer[16];
			std::swprintf(buffer, 16, L"Size: %1.2f", mp_light->GetSize());
			mp_lSize->SetCaption(buffer);
		}
		void SpotLightEditor::WriteAngle()
		{
			wchar_t buffer[32];
			std::swprintf(buffer, 32, L"Beam angle: %1.2f", mp_light->GetBeamAngle());
			mp_lAngle->SetCaption(buffer);
		}

		void SpotLightEditor::TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_tbPosX->GetPosition() / 10.0f,
				mp_light->GetPosition().y,
				mp_light->GetPosition().z));
			WritePosition(mp_light->GetPosition());
		}
		void SpotLightEditor::TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_light->GetPosition().x,
				mp_tbPosY->GetPosition() / 10.0f,
				mp_light->GetPosition().z));
			WritePosition(mp_light->GetPosition());
		}
		void SpotLightEditor::TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetPosition(Math::vec3<float>(
				mp_light->GetPosition().x,
				mp_light->GetPosition().y,
				mp_tbPosZ->GetPosition() / 10.0f));
			WritePosition(mp_light->GetPosition());
		}

		void SpotLightEditor::TBPhi_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			phi = mp_tbPhi->GetPosition() / 100.0f;
			Math::vec3<float> direction(cosf(theta) * cosf(phi), sinf(theta), cosf(theta) * sinf(phi));
			mp_light->SetDirection(direction);
			WriteDirection();
		}
		void SpotLightEditor::TBTheta_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			theta = mp_tbTheta->GetPosition() / 100.0f;
			Math::vec3<float> direction(cosf(theta) * cosf(phi), sinf(theta), cosf(theta) * sinf(phi));
			mp_light->SetDirection(direction);
			WriteDirection();
		}

		void SpotLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetSize(mp_tbSize->GetPosition() / 100.0f);
			WriteSize();
		}
		void SpotLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				mp_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				mp_light->SetEmission(1.0f);
			}
		}
		void SpotLightEditor::TBAngle_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetBeamAngle(mp_tbAngle->GetPosition() / 100.0f);
			WriteAngle();
		}

		// ~~~~~~~~ [CLASS] DirectLightEditor ~~~~~~~~
		DirectLightEditor::DirectLightEditor(WAF::Window* window, RZ::DirectLight* light)
			: mp_window(window)
			, mp_light(light)
			, m_mode(Mode::ByAxes)
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 300), L"Direct light properties"));

			mp_lDirection = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 100, 100, 30), L"Direction", WAF::Label::TextAlignment::Center));
			mp_bMode = mp_window->CreateChild(WAF::ConStruct<WAF::Button>(
				WAF::Rect(120, 100, 80, 30), L"Angular Mode"));
			mp_bMode->BindEventFunc(&DirectLightEditor::BMode_OnClick, this);

			// direction by axes
			mp_lDirX = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 155, 50, 20), L"X:", WAF::Label::TextAlignment::Center));
			mp_tbDirX = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 140, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetDirection().x * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbDirX->BindEventFunc(&DirectLightEditor::TBDirectionX_OnDrag, this);

			mp_lDirY = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 195, 50, 20), L"Y:", WAF::Label::TextAlignment::Center));
			mp_tbDirY = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 180, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetDirection().y * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbDirY->BindEventFunc(&DirectLightEditor::TBDirectionY_OnDrag, this);

			mp_lDirZ = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 235, 50, 20), L"Z:", WAF::Label::TextAlignment::Center));
			mp_tbDirZ = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 220, 200, 40),
				WAF::Range(-100, 100),
				mp_light->GetDirection().z * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbDirZ->BindEventFunc(&DirectLightEditor::TBDirectionZ_OnDrag, this);

			// direction by angles
			mp_lPhi = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 155, 60, 20), L"Phi:", WAF::Label::TextAlignment::Center));
			mp_tbPhi = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(80, 140, 200, 40),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbPhi->BindEventFunc(&DirectLightEditor::TBDirectionPhi_OnDrag, this);

			mp_lTheta = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 195, 60, 20), L"Theta:", WAF::Label::TextAlignment::Center));
			mp_tbTheta = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(80, 180, 200, 40),
				WAF::Range(-157, 157),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				30u, false));
			mp_tbTheta->BindEventFunc(&DirectLightEditor::TBDirectionTheta_OnDrag, this);

			mp_lPhi->Hide();
			mp_tbPhi->Hide();
			mp_lTheta->Hide();
			mp_tbTheta->Hide();


			// size
			mp_lSize = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 295, 60, 20),
				L"Size: "));
			mp_tbSize = mp_window->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 280, 200, 40),
				WAF::Range(1, 314),
				mp_light->GetAngularSize() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbSize->BindEventFunc(&DirectLightEditor::TBSize_OnDrag, this);
			WriteSize();

			// emission
			mp_lEmission = mp_window->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(20, 330, 60, 20),
				L"Emission: "));
			mp_eEmission = mp_window->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(80, 330, 100, 20), std::to_wstring(int(mp_light->GetEmission())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&DirectLightEditor::EditEmission_OnInput, this);
		}
		DirectLightEditor::~DirectLightEditor()
		{
			mp_gbProperties->Destroy();

			mp_lDirection->Destroy();
			mp_bMode->Destroy();

			mp_lDirX->Destroy();
			mp_tbDirX->Destroy();
			mp_lDirY->Destroy();
			mp_tbDirY->Destroy();
			mp_lDirZ->Destroy();
			mp_tbDirZ->Destroy();

			mp_lPhi->Destroy();
			mp_tbPhi->Destroy();
			mp_lTheta->Destroy();
			mp_tbTheta->Destroy();

			mp_lSize->Destroy();
			mp_tbSize->Destroy();
			mp_lEmission->Destroy();
			mp_eEmission->Destroy();
		}

		void DirectLightEditor::UpdateState()
		{
			if (m_mode == Mode::ByAxes)
			{
				mp_tbDirX->SetThumbPosition(mp_light->GetDirection().x * 100.0f);
				mp_tbDirY->SetThumbPosition(mp_light->GetDirection().y * 100.0f);
				mp_tbDirZ->SetThumbPosition(mp_light->GetDirection().z * 100.0f);
			}
			else if (m_mode == Mode::ByAngles)
			{
				phi = atan2f(mp_light->GetDirection().z, mp_light->GetDirection().x);
				mp_tbPhi->SetThumbPosition(phi * 100.0f);

				theta = asinf(mp_light->GetDirection().y);
				mp_tbTheta->SetThumbPosition(theta * 100.0f);
			}

			WriteDirection();
		}

		void DirectLightEditor::WriteDirection()
		{
			if (m_mode == Mode::ByAxes)
			{
				wchar_t buffer[10];

				std::swprintf(buffer, 10, L"X: %1.2f", mp_light->GetDirection().x);
				mp_lDirX->SetCaption(buffer);
				std::swprintf(buffer, 10, L"Y: %1.2f", mp_light->GetDirection().y);
				mp_lDirY->SetCaption(buffer);
				std::swprintf(buffer, 10, L"Z: %1.2f", mp_light->GetDirection().z);
				mp_lDirZ->SetCaption(buffer);
			}
			else if (m_mode == Mode::ByAngles)
			{
				wchar_t buffer[32];

				std::swprintf(buffer, 32, L"Phi: %1.2f", phi);
				mp_lPhi->SetCaption(buffer);
				std::swprintf(buffer, 32, L"Theta: %1.2f", theta);
				mp_lTheta->SetCaption(buffer);
			}
		}
		void DirectLightEditor::WriteSize()
		{
			wchar_t buffer[16];
			std::swprintf(buffer, 16, L"Size: %1.2f", mp_light->GetAngularSize());
			mp_lSize->SetCaption(buffer);
		}
		
		void DirectLightEditor::BMode_OnClick(WAF::Button::Events::EventClick& event)
		{
			if (m_mode == Mode::ByAxes)
			{
				m_mode = Mode::ByAngles;
				mp_bMode->SetCaption(L"Axes Mode");

				mp_lDirX->Hide();
				mp_tbDirX->Hide();
				mp_lDirY->Hide();
				mp_tbDirY->Hide();
				mp_lDirZ->Hide();
				mp_tbDirZ->Hide();

				mp_lPhi->Show();
				mp_tbPhi->Show();
				mp_lTheta->Show();
				mp_tbTheta->Show();
			}
			else
			{
				m_mode = Mode::ByAxes;
				mp_bMode->SetCaption(L"Angles Mode");

				mp_lDirX->Show();
				mp_tbDirX->Show();
				mp_lDirY->Show();
				mp_tbDirY->Show();
				mp_lDirZ->Show();
				mp_tbDirZ->Show();

				mp_lPhi->Hide();
				mp_tbPhi->Hide();
				mp_lTheta->Hide();
				mp_tbTheta->Hide();
			}

			UpdateState();
		}

		void DirectLightEditor::TBDirectionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetDirection(Math::vec3<float>(
				mp_tbDirX->GetPosition() / 100.0f,
				mp_light->GetDirection().y,
				mp_light->GetDirection().z));

			UpdateState();
		}
		void DirectLightEditor::TBDirectionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetDirection(Math::vec3<float>(
				mp_light->GetDirection().x,
				mp_tbDirY->GetPosition() / 100.0f,
				mp_light->GetDirection().z));

			UpdateState();
		}
		void DirectLightEditor::TBDirectionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetDirection(Math::vec3<float>(
				mp_light->GetDirection().x,
				mp_light->GetDirection().y,
				mp_tbDirZ->GetPosition() / 100.0f));

			UpdateState();
		}
		void DirectLightEditor::TBDirectionPhi_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			phi = mp_tbPhi->GetPosition() / 100.0f;
			Math::vec3<float> direction(cosf(theta) * cosf(phi), sinf(theta), cosf(theta) * sinf(phi));
			mp_light->SetDirection(direction);
			WriteDirection();
		}
		void DirectLightEditor::TBDirectionTheta_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			theta = mp_tbTheta->GetPosition() / 100.0f;
			Math::vec3<float> direction(cosf(theta) * cosf(phi), sinf(theta), cosf(theta) * sinf(phi));
			mp_light->SetDirection(direction);
			WriteDirection();
		}

		void DirectLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			mp_light->SetAngularSize(mp_tbSize->GetPosition() / 100.0f);
			WriteSize();
		}
		void DirectLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				mp_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				mp_light->SetEmission(1.0f);
			}
		}

		// ~~~~~~~~ [CLASS] SphereEditor ~~~~~~~~
		SphereEditor::SphereEditor(WAF::Window* window, RZ::Sphere* sphere)
			: mp_window(window)
			, mp_sphere(sphere)
			, m_position_editor(mp_window, mp_sphere, WAF::Point(20, 100))
			, m_rotation_editor(mp_window, mp_sphere, WAF::Point(20, 260))
			, m_scale_editor(mp_window, mp_sphere, WAF::Point(20, 420))
			, m_material_editor(mp_window, mp_sphere, WAF::Point(20, 580))
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 760), L"Sphere properties"));
		}
		SphereEditor::~SphereEditor()
		{
			mp_gbProperties->Destroy();
		}



		// ~~~~~~~~ [CLASS] MeshEditor ~~~~~~~~
		MeshEditor::MeshEditor(WAF::Window* window, RZ::Mesh* mesh)
			: mp_window(window)
			, mp_mesh(mesh)
			, m_position_editor(mp_window, mp_mesh, WAF::Point(20, 100))
			, m_rotation_editor(mp_window, mp_mesh, WAF::Point(20, 260))
			, m_scale_editor(mp_window, mp_mesh, WAF::Point(20, 420))
			, m_material_editor(mp_window, mp_mesh, WAF::Point(20, 580))
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 760), L"Mesh properties"));
		}
		MeshEditor::~MeshEditor()
		{
			mp_gbProperties->Destroy();
		}
	}
}