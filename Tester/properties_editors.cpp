#include "properties_editors.h"
#include "application.h"

#include <string>

namespace Tester
{
	namespace UI
	{		
		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		PositionEditor::PositionEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Math::vec3f&)> function,
			const Math::vec3f& initial_position)
			: m_notify_function(function)
			, m_position(initial_position)
		{
			// panel
			mp_pPosition = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 120)));
			// label
			mp_lPosition = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"position", WAF::Label::TextAlignment::Center));

			// position x
			mp_lPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 20), L"X:"));
			mp_tbPosX = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 20, 200, 30),
				WAF::Range(-1000, 1000),
				m_position.x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbPosX->BindEventFunc(&PositionEditor::TBPositionX_OnDrag, this);

			// position y
			mp_lPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 65, 50, 20), L"Y:"));
			mp_tbPosY = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 50, 200, 30),
				WAF::Range(-1000, 1000),
				m_position.y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbPosY->BindEventFunc(&PositionEditor::TBPositionY_OnDrag, this);

			// position z
			mp_lPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 95, 50, 20), L"Z:"));
			mp_tbPosZ = mp_pPosition->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 80, 200, 30),
				WAF::Range(-1000, 1000),
				m_position.z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbPosZ->BindEventFunc(&PositionEditor::TBPositionZ_OnDrag, this);

			WritePosition();
		}
		PositionEditor::~PositionEditor()
		{
			mp_pPosition->Destroy();
		}

		void PositionEditor::SetPosition(const Math::vec3f& position)
		{
			m_position = position;

			mp_tbPosX->SetThumbPosition(m_position.x * 100.0f);
			mp_tbPosY->SetThumbPosition(m_position.y * 100.0f);
			mp_tbPosZ->SetThumbPosition(m_position.z * 100.0f);

			WritePosition();
		}
		void PositionEditor::WritePosition()
		{
			const int buff_size = 16;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"X: %1.2f", m_position.x);
			mp_lPosX->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Y: %1.2f", m_position.y);
			mp_lPosY->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Z: %1.2f", m_position.z);
			mp_lPosZ->SetCaption(buffer);
		}
		void PositionEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_position);
		}

		void PositionEditor::TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_position.x = mp_tbPosX->GetPosition() / 100.0f;
			WritePosition();
			Notify();
		}
		void PositionEditor::TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_position.y = mp_tbPosY->GetPosition() / 100.0f;
			WritePosition();
			Notify();
		}
		void PositionEditor::TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_position.z = mp_tbPosZ->GetPosition() / 100.0f;
			WritePosition();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		RotationEditor::RotationEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Math::vec3f&)> function,
			const Math::vec3f& initial_rotation)
			: m_notify_function(function)
			, m_rotation(initial_rotation)
		{
			// panel and caption
			mp_pRotation = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 120)));
			mp_lRotation = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"rotation", WAF::Label::TextAlignment::Center));

			// rotation x
			mp_lRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 20), L"X:"));
			mp_tbRotX = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 20, 200, 30),
				WAF::Range(-314, 314),
				m_rotation.x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbRotX->BindEventFunc(&RotationEditor::TBRotationX_OnDrag, this);

			mp_lRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 65, 50, 20), L"Y:"));
			mp_tbRotY = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 50, 200, 30),
				WAF::Range(-314, 314),
				m_rotation.y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbRotY->BindEventFunc(&RotationEditor::TBRotationY_OnDrag, this);

			mp_lRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 95, 50, 20), L"Z:"));
			mp_tbRotZ = mp_pRotation->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 80, 200, 30),
				WAF::Range(-314, 314),
				m_rotation.z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbRotZ->BindEventFunc(&RotationEditor::TBRotationZ_OnDrag, this);

			WriteRotation();
		}
		RotationEditor::~RotationEditor()
		{
			mp_pRotation->Destroy();
		}

		void RotationEditor::SetRotation(const Math::vec3f& rotation)
		{
			m_rotation = rotation;

			mp_tbRotX->SetThumbPosition(m_rotation.x * 100.0f);
			mp_tbRotY->SetThumbPosition(m_rotation.y * 100.0f);
			mp_tbRotZ->SetThumbPosition(m_rotation.z * 100.0f);

			WriteRotation();
		}
		void RotationEditor::WriteRotation()
		{
			const int buff_size = 16;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"X: %1.2f", m_rotation.x);
			mp_lRotX->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Y: %1.2f", m_rotation.y);
			mp_lRotY->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Z: %1.2f", m_rotation.z);
			mp_lRotZ->SetCaption(buffer);
		}
		void RotationEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_rotation);
		}

		void RotationEditor::TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_rotation.x = mp_tbRotX->GetPosition() / 100.0f;
			WriteRotation();
			Notify();
		}
		void RotationEditor::TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_rotation.y = mp_tbRotY->GetPosition() / 100.0f;
			WriteRotation();
			Notify();
		}
		void RotationEditor::TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_rotation.z = mp_tbRotZ->GetPosition() / 100.0f;
			WriteRotation();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] PositionEditor ~~~~~~~~
		ScaleEditor::ScaleEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Math::vec3f&)> function,
			const Math::vec3f& initial_scale)
			: m_notify_function(function)
			, m_scale(initial_scale)
		{
			// panel and label
			mp_pScale = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 120)));
			mp_lScale = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"scale:", WAF::Label::TextAlignment::Center));

			mp_lScaleX = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 20), L"X:"));
			mp_tbScaleX = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 20, 200, 30),
				WAF::Range(1, 500),
				m_scale.x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbScaleX->BindEventFunc(&ScaleEditor::TBScaleX_OnDrag, this);

			mp_lScaleY = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 65, 50, 20), L"Y:"));
			mp_tbScaleY = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 50, 200, 30),
				WAF::Range(1, 500),
				m_scale.y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbScaleY->BindEventFunc(&ScaleEditor::TBScaleY_OnDrag, this);

			mp_lScaleZ = mp_pScale->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 95, 50, 20), L"Z:"));
			mp_tbScaleZ = mp_pScale->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 80, 200, 30),
				WAF::Range(1, 500),
				m_scale.z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbScaleZ->BindEventFunc(&ScaleEditor::TBScaleZ_OnDrag, this);

			WriteScale();
		}
		ScaleEditor::~ScaleEditor()
		{
			mp_pScale->Destroy();
		}

		void ScaleEditor::WriteScale()
		{
			const int buff_size = 16;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"X: %1.2f", m_scale.x);
			mp_lScaleX->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Y: %1.2f", m_scale.y);
			mp_lScaleY->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Z: %1.2f", m_scale.z);
			mp_lScaleZ->SetCaption(buffer);
		}
		void ScaleEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_scale);
		}

		void ScaleEditor::TBScaleX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_scale.x = mp_tbScaleX->GetPosition() / 100.0f;
			WriteScale();
			Notify();
		}
		void ScaleEditor::TBScaleY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_scale.y = mp_tbScaleY->GetPosition() / 100.0f;
			WriteScale();
			Notify();
		}
		void ScaleEditor::TBScaleZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_scale.z = mp_tbScaleZ->GetPosition() / 100.0f;
			WriteScale();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// ~~~~~~~~ [STRUCT] CenterEditor ~~~~~~~~
		CenterEditor::CenterEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Math::vec3f&)> function,
			const Math::vec3f& initial_center)
			: m_notify_function(function)
			, m_center(initial_center)
		{
			// panel and label
			mp_pCenter = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 120)));
			mp_lCenter = mp_pCenter->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"center:", WAF::Label::TextAlignment::Center));

			mp_lCenterX = mp_pCenter->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 35, 50, 20), L"X:"));
			mp_tbCenterX = mp_pCenter->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 20, 200, 30),
				WAF::Range(-500, 500),
				m_center.x * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbCenterX->BindEventFunc(&CenterEditor::TBCenterX_OnDrag, this);

			mp_lCenterY = mp_pCenter->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 65, 50, 20), L"Y:"));
			mp_tbCenterY = mp_pCenter->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 50, 200, 30),
				WAF::Range(-500, 500),
				m_center.y * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbCenterY->BindEventFunc(&CenterEditor::TBCenterY_OnDrag, this);

			mp_lCenterZ = mp_pCenter->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 95, 50, 20), L"Z:"));
			mp_tbCenterZ = mp_pCenter->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 80, 200, 30),
				WAF::Range(-500, 500),
				m_center.z * 100.0f,
				10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				100u, false));
			mp_tbCenterZ->BindEventFunc(&CenterEditor::TBCenterZ_OnDrag, this);

			WriteCenter();
		}
		CenterEditor::~CenterEditor()
		{
			mp_pCenter->Destroy();
		}

		void CenterEditor::WriteCenter()
		{
			const int buff_size = 16;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"X: %1.2f", m_center.x);
			mp_lCenterX->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Y: %1.2f", m_center.y);
			mp_lCenterY->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Z: %1.2f", m_center.z);
			mp_lCenterZ->SetCaption(buffer);
		}
		void CenterEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_center);
		}

		void CenterEditor::TBCenterX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_center.x = mp_tbCenterX->GetPosition() / 100.0f;
			WriteCenter();
			Notify();
		}
		void CenterEditor::TBCenterY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_center.y = mp_tbCenterY->GetPosition() / 100.0f;
			WriteCenter();
			Notify();
		}
		void CenterEditor::TBCenterZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_center.z = mp_tbCenterZ->GetPosition() / 100.0f;
			WriteCenter();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		// ~~~~~~~~ [STRUCT] DirectionEditor ~~~~~~~~
		DirectionEditor::DirectionEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Math::vec3f&)> function,
			const Math::vec3f& initial_direction)
			: m_notify_function(function)
			, m_direction(initial_direction)
		{
			// panel and label
			mp_pDirection = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 140)));
			mp_lDirection = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"direction", WAF::Label::TextAlignment::Center));

			mp_bMode = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Button>(
				WAF::Rect(5, 20, 80, 20), L"Angular Mode"));
			mp_bMode->BindEventFunc(&DirectionEditor::BMode_OnClick, this);

			// direction by axes
			mp_lDirX = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 55, 50, 20), L"X:", WAF::Label::TextAlignment::Left));
			mp_tbDirX = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 40, 200, 30),
				WAF::Range(-100, 100),
				m_direction.x * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				10u, false));
			mp_tbDirX->BindEventFunc(&DirectionEditor::TBDirectionX_OnDrag, this);

			mp_lDirY = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 85, 50, 20), L"Y:", WAF::Label::TextAlignment::Left));
			mp_tbDirY = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 70, 200, 30),
				WAF::Range(-100, 100),
				m_direction.y * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				10u, false));
			mp_tbDirY->BindEventFunc(&DirectionEditor::TBDirectionY_OnDrag, this);

			mp_lDirZ = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 115, 50, 20), L"Z:", WAF::Label::TextAlignment::Left));
			mp_tbDirZ = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 100, 200, 30),
				WAF::Range(-100, 100),
				m_direction.z * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				10u, false));
			mp_tbDirZ->BindEventFunc(&DirectionEditor::TBDirectionZ_OnDrag, this);

			// direction by angles
			mp_lPhi = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 55, 60, 20), L"Phi:", WAF::Label::TextAlignment::Left));
			mp_tbPhi = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(80, 40, 170, 30),
				WAF::Range(-314, 314),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				30u, false));
			mp_tbPhi->BindEventFunc(&DirectionEditor::TBDirectionPhi_OnDrag, this);

			mp_lTheta = mp_pDirection->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 85, 60, 20), L"Theta:", WAF::Label::TextAlignment::Left));
			mp_tbTheta = mp_pDirection->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(80, 70, 170, 30),
				WAF::Range(-157, 157),
				0, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				30u, false));
			mp_tbTheta->BindEventFunc(&DirectionEditor::TBDirectionTheta_OnDrag, this);

			mp_lPhi->Hide();
			mp_tbPhi->Hide();
			mp_lTheta->Hide();
			mp_tbTheta->Hide();

			WriteDirection();
		}
		DirectionEditor::~DirectionEditor()
		{
			mp_pDirection->Destroy();
		}

		void DirectionEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_direction);
		}
		void DirectionEditor::UpdateState()
		{
			m_direction.Normalize();
			if (m_mode == Mode::ByAxes)
			{
				mp_tbDirX->SetThumbPosition(m_direction.x * 100.0f);
				mp_tbDirY->SetThumbPosition(m_direction.y * 100.0f);
				mp_tbDirZ->SetThumbPosition(m_direction.z * 100.0f);
			}
			else if (m_mode == Mode::ByAngles)
			{
				m_phi = atan2f(m_direction.z, m_direction.x);
				mp_tbPhi->SetThumbPosition(m_phi * 100.0f);

				m_theta = asinf(m_direction.y);
				mp_tbTheta->SetThumbPosition(m_theta * 100.0f);
			}

			WriteDirection();
		}
		void DirectionEditor::WriteDirection()
		{
			if (m_mode == Mode::ByAxes)
			{
				wchar_t buffer[10];

				std::swprintf(buffer, 10, L"X: %1.2f", m_direction.x);
				mp_lDirX->SetCaption(buffer);
				std::swprintf(buffer, 10, L"Y: %1.2f", m_direction.y);
				mp_lDirY->SetCaption(buffer);
				std::swprintf(buffer, 10, L"Z: %1.2f", m_direction.z);
				mp_lDirZ->SetCaption(buffer);
			}
			else if (m_mode == Mode::ByAngles)
			{
				wchar_t buffer[32];

				std::swprintf(buffer, 32, L"Phi: %1.2f", m_phi);
				mp_lPhi->SetCaption(buffer);
				std::swprintf(buffer, 32, L"Theta: %1.2f", m_theta);
				mp_lTheta->SetCaption(buffer);
			}
		}
		void DirectionEditor::BMode_OnClick(WAF::Button::Events::EventClick& event)
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
		void DirectionEditor::TBDirectionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_direction.x = mp_tbDirX->GetPosition() / 100.0f;
			UpdateState();
			Notify();
		}
		void DirectionEditor::TBDirectionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_direction.y = mp_tbDirY->GetPosition() / 100.0f;
			UpdateState();
			Notify();
		}
		void DirectionEditor::TBDirectionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_direction.z = mp_tbDirZ->GetPosition() / 100.0f;
			UpdateState();
			Notify();
		}
		void DirectionEditor::TBDirectionPhi_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_phi = mp_tbPhi->GetPosition() / 100.0f;
			m_direction = Math::vec3f(cosf(m_theta) * cosf(m_phi), sinf(m_theta), cosf(m_theta) * sinf(m_phi));
			UpdateState();
			Notify();
		}
		void DirectionEditor::TBDirectionTheta_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_theta = mp_tbTheta->GetPosition() / 100.0f;
			m_direction = Math::vec3f(cosf(m_theta) * cosf(m_phi), sinf(m_theta), cosf(m_theta) * sinf(m_phi));
			UpdateState();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
		
		// ~~~~~~~~ [STRUCT] MaterialEditor ~~~~~~~~
		MaterialEditor::MaterialEditor(
			WAF::Window* window, 
			const RZ::Handle<RZ::Material>& material,
			const WAF::Point& position)
			: mp_window(window)
			, m_material(material)
		{
			if (!m_material) return;

			// panel and header
			mp_pMaterial = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 350)));
			mp_lMaterial = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 0, 260, 15), L"material:", WAF::Label::TextAlignment::Center));

			// metalness
			mp_lMetalness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 25, 100, 15), L"Metalness:"));
			mp_tbMetalness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 40, 245, 25),
				WAF::Range(0, 100),
				m_material->GetMetalic() * 100.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbMetalness->BindEventFunc(&MaterialEditor::TBMetalic_OnDrag, this);

			// specularity
			mp_lSpecularity = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 75, 100, 15), L"Specularity:"));
			mp_tbSpecularity = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 90, 245, 25),
				WAF::Range(0, 100),
				m_material->GetSpecular() * 100.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbSpecularity->BindEventFunc(&MaterialEditor::TBSpecular_OnDrag, this);

			// roughness
			mp_lRoughness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 125, 100, 15), L"Roughness:"));
			mp_tbRoughness = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 140, 245, 25),
				WAF::Range(0, 100),
				m_material->GetRoughness() * 100.0f,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbRoughness->BindEventFunc(&MaterialEditor::TBRoughness_OnDrag, this);

			// opacity
			mp_lOpacity = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 175, 150, 15), L"Opacity:"));
			mp_tbOpacity = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 190, 245, 25),
				WAF::Range(0, 255),
				m_material->GetColor().alpha,
				1u, 16u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbOpacity->BindEventFunc(&MaterialEditor::TBTransmission_OnDrag, this);

			// IOR
			mp_lIOR = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 225, 150, 15), L"Refraction index:"));
			mp_tbIOR = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 240, 245, 25),
				WAF::Range(100, 500),
				m_material->GetIOR() * 100.0f,
				1u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				100u, false));
			mp_tbIOR->BindEventFunc(&MaterialEditor::TBIOR_OnDrag, this);

			// Scattering
			mp_lScattering = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 275, 50, 15), L"Scattering:"));
			mp_tbScattering = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 290, 245, 25),
				WAF::Range(0, 500),
				m_material->GetScattering() * 100.0f,
				1u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				100u, false));
			mp_tbScattering->BindEventFunc(&MaterialEditor::TBScattering_OnDrag, this);

			// Emittance
			mp_lEmission = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 325, 50, 15), L"Emission:"));
			mp_eEmission = mp_pMaterial->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(60, 323, 100, 20), std::to_wstring(int(m_material->GetEmission())),
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

			std::swprintf(buffer, buff_size, L"Metalic: %1.2f", m_material->GetMetalic());
			mp_lMetalness->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Specularity: %1.2f", m_material->GetSpecular());
			mp_lSpecularity->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Roughness: %1.4f", m_material->GetRoughness());
			mp_lRoughness ->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Opacity: %1.2f", m_material->GetColor().alpha / 255.0f);
			mp_lOpacity->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Refraction index: %1.2f", m_material->GetIOR());
			mp_lIOR->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Scattering: %1.2f", m_material->GetScattering());
			mp_lScattering->SetCaption(buffer);
		}

		void MaterialEditor::TBMetalic_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_material->SetMetalic(
				mp_tbMetalness->GetPosition() /
				static_cast<float>(mp_tbMetalness->GetMaxTrackValue()));
			WriteMaterialProps();
		}
		void MaterialEditor::TBSpecular_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_material->SetSpecular(
				mp_tbSpecularity->GetPosition() / 
				static_cast<float>(mp_tbSpecularity->GetMaxTrackValue()));
			WriteMaterialProps();
		}
		void MaterialEditor::TBRoughness_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			const float r = mp_tbRoughness->GetPosition() / 100.0f;
			m_material->SetRoughness(
				r * r);
			WriteMaterialProps();
		}
		void MaterialEditor::TBTransmission_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			Graphics::Color c = m_material->GetColor();
			c.alpha = mp_tbOpacity->GetPosition();
			m_material->SetColor(c);
			WriteMaterialProps();
		}
		void MaterialEditor::TBIOR_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_material->SetIOR(
				mp_tbIOR->GetPosition() / 100.0f);
			WriteMaterialProps();
		}
		void MaterialEditor::TBScattering_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_material->SetScattering(
				mp_tbScattering->GetPosition() / 100.0f);
			WriteMaterialProps();
		}
		void MaterialEditor::EEmission_OnEdit(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				m_material->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				m_material->SetEmission(0.0f);
			}
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [STRUCT] ColorEditor ~~~~~~~~
		ColorEditor::ColorEditor(
			WAF::Window* window,
			const WAF::Point& position,
			const std::function<void(const Graphics::Color&)> function,
			const Graphics::Color& initial_color)
			: m_notify_function(function)
			, m_color(initial_color)
		{
			mp_pColor = window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(position.x, position.y, 260, 170)));
			mp_lColor = mp_pColor->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(0, 5, 260, 15), L"color", WAF::Label::TextAlignment::Center));

			// hue
			mp_lHue = mp_pColor->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 15, 100, 15), L"Hue:"));
			mp_tbHue = mp_pColor->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 30, 245, 25),
				WAF::Range(0, 360),
				0u, 10u, 60u, 
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbHue->BindEventFunc(&ColorEditor::TBHue_OnDrag, this);

			// saturation
			mp_lSaturation = mp_pColor->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 65, 100, 15), L"Saturation:"));
			mp_tbSaturation = mp_pColor->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 80, 245, 25),
				WAF::Range(0, 100),
				100u, 
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbSaturation->BindEventFunc(&ColorEditor::TBSaturation_OnDrag, this);

			// value
			mp_lValue = mp_pColor->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(10, 115, 100, 15), L"Value:"));
			mp_tbValue = mp_pColor->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(5, 130, 245, 25),
				WAF::Range(0, 100),
				100u,
				1u, 10u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Default,
				10u, false));
			mp_tbValue->BindEventFunc(&ColorEditor::TBValue_OnDrag, this);

			WriteColor();
		}
		ColorEditor::~ColorEditor()
		{
			mp_pColor->Destroy();
		}

		void ColorEditor::Notify()
		{
			if (m_notify_function) m_notify_function(m_color);
		}
		void ColorEditor::WriteColor()
		{
			const size_t buff_size = 32u;
			wchar_t buffer[buff_size];

			std::swprintf(buffer, buff_size, L"Hue: %d", mp_tbHue->GetPosition());
			mp_lHue->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Saturation: %d", mp_tbSaturation->GetPosition());
			mp_lSaturation->SetCaption(buffer);
			std::swprintf(buffer, buff_size, L"Value: %d", mp_tbValue->GetPosition());
			mp_lValue->SetCaption(buffer);
		}

		void ColorEditor::HSVtoRGB()
		{
			const float H = mp_tbHue->GetPosition();
			const float S = mp_tbSaturation->GetPosition();
			const float V = mp_tbValue->GetPosition();

			const float s = S / 100.0f;
			const float v = V / 100.0f;
			const float C = s * v;
			const float X = C * (1.0f - abs(fmodf(H / 60.0f, 2) - 1.0f));
			const float m = v - C;

			float r, g, b;
			if (H >= 0.0f && H < 60.0f) 
				r = C, g = X, b = 0.0f;
			else if (H >= 60.0f && H < 120.0f)
				r = X, g = C, b = 0.0f;
			else if (H >= 120.0f && H < 180.0f)
				r = 0.0f, g = C, b = X;
			else if (H >= 180.0f && H < 240.0f)
				r = 0.0f, g = X, b = C;
			else if (H >= 240.0f && H < 300.0f)
				r = X, g = 0.0f, b = C;
			else
				r = C, g = 0.0f, b = X;

			const uint8_t R = (r + m) * 255u;
			const uint8_t G = (g + m) * 255u;
			const uint8_t B = (b + m) * 255u;

			m_color = Graphics::Color(R, G, B);
		}

		void ColorEditor::TBHue_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			WriteColor();
			HSVtoRGB();
			Notify();
		}
		void ColorEditor::TBSaturation_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			WriteColor();
			HSVtoRGB();
			Notify();
		}
		void ColorEditor::TBValue_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			WriteColor();
			HSVtoRGB();
			Notify();
		}
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		// ~~~~~~~~ [CLASS] CameraPropsEditor ~~~~~~~~
		CameraPropsEditor::CameraPropsEditor(
			WAF::Window* window, 
			const RZ::Handle<RZ::Camera>& camera)
			: mp_window(window)
			, m_camera(camera)
			, m_pos_editor(
				mp_window, 
				WAF::Point(20, 100),
				std::bind(&CameraPropsEditor::NotifyPosition, this, std::placeholders::_1),
				camera->GetPosition())
			, m_rot_editor(
				mp_window,
				WAF::Point(20, 230),
				std::bind(&CameraPropsEditor::NotifyRotation, this, std::placeholders::_1),
				camera->GetRotation())
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 500), L"Camera properties"));

			// other panel
			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 360, 260, 100)));

			// fov
			mp_lFov = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 50, 20), L"Fov: "));
			mp_tbFov = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 0, 200, 30),
				WAF::Range(10, 300),
				m_camera->GetFov().value() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				30u, false));
			mp_tbFov->BindEventFunc(&CameraPropsEditor::TBFov_OnDrag, this);

			// focal distance
			mp_lFocalDistance = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 40, 50, 30), L"Focal distance: "));
			mp_tbFocalDistance = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 30, 200, 30),
				WAF::Range(10, 3000),
				m_camera->GetFocalDistance() * 100.0f, 10u, 50u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				200u, false));
			mp_tbFocalDistance->BindEventFunc(&CameraPropsEditor::TBFocalDistance_OnDrag, this);

			// aperature
			mp_lAperature = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 75, 50, 20), L"Aperature: "));
			mp_tbAperature = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(55, 60, 200, 30),
				WAF::Range(0, 100),
				m_camera->GetAperture() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Top,
				30u, false));
			mp_tbAperature->BindEventFunc(&CameraPropsEditor::TBAperature_OnDrag, this);
		}
		CameraPropsEditor::~CameraPropsEditor()
		{
			mp_gbProperties->Destroy();
			mp_pOthers->Destroy();
		}

		void CameraPropsEditor::UpdateState()
		{
			m_pos_editor.SetPosition(m_camera->GetPosition());
			m_rot_editor.SetRotation(m_camera->GetRotation());
		}

		void CameraPropsEditor::NotifyPosition(const Math::vec3f& position)
		{
			m_camera->SetPosition(position);
		}
		void CameraPropsEditor::NotifyRotation(const Math::vec3f& rotation)
		{
			m_camera->SetRotation(rotation);
		}

		void CameraPropsEditor::TBFov_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_camera->SetFov(mp_tbFov->GetPosition() / 100.0f);
		}
		void CameraPropsEditor::TBFocalDistance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_camera->SetFocalDistance(mp_tbFocalDistance->GetPosition() / 100.0f);
		}
		void CameraPropsEditor::TBAperature_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_camera->SetAperture(mp_tbAperature->GetPosition() / 100.0f);
		}



		// ~~~~~~~~ [CLASS] PointLightEditor ~~~~~~~~
		PointLightEditor::PointLightEditor(
			WAF::Window* window,
			const RZ::Handle<RZ::PointLight>& light)
			: mp_window(window)
			, m_light(light)
			, m_position_editor(
				mp_window,
				WAF::Point(20, 100),
				std::bind(&PointLightEditor::NotifyPosition, this, std::placeholders::_1),
				light->GetPosition())
			, m_color_editor(
				mp_window,
				WAF::Point(20, 230),
				std::bind(&PointLightEditor::NotifyColor, this, std::placeholders::_1),
				light->GetColor())
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 500), L"Point light properties"));

			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 410, 260, 70)));

			// size
			mp_lSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 40, 15),
				L"Size: "));
			mp_tbSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(50, 0, 200, 40),
				WAF::Range(1, 100),
				m_light->GetSize() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				50u, false));
			mp_tbSize->BindEventFunc(&PointLightEditor::TBSize_OnDrag, this);

			// emission
			mp_lEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 45, 40, 15),
				L"Emission: "));
			mp_eEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(60, 43, 100, 20), std::to_wstring(int(m_light->GetEmission())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&PointLightEditor::EditEmission_OnInput, this);
		}
		PointLightEditor::~PointLightEditor()
		{
			mp_gbProperties->Destroy();
			mp_pOthers->Destroy();
		}

		void PointLightEditor::NotifyPosition(const Math::vec3f& position)
		{
			m_light->SetPosition(position);
		}
		void PointLightEditor::NotifyColor(const Graphics::Color& color)
		{
			m_light->SetColor(color);
		}

		void PointLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_light->SetSize(mp_tbSize->GetPosition() / 100.0f);
		}
		void PointLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				m_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				m_light->SetEmission(1.0f);
			}
		}



		// ~~~~~~~~ [CLASS] SpotLightEditor ~~~~~~~~
		SpotLightEditor::SpotLightEditor(
			WAF::Window* window,
			const RZ::Handle<RZ::SpotLight>& light)
			: mp_window(window)
			, m_light(light)
			, m_position_editor(
				mp_window,
				WAF::Point(20, 100),
				std::bind(&SpotLightEditor::NotifyPosition, this, std::placeholders::_1),
				light->GetPosition())
			, m_direction_editor(
				mp_window,
				WAF::Point(20, 230),
				std::bind(&SpotLightEditor::NotifyDirection, this, std::placeholders::_1),
				m_light->GetDirection())
			, m_color_editor(
				mp_window,
				WAF::Point(20, 380),
				std::bind(&SpotLightEditor::NotifyColor, this, std::placeholders::_1),
				m_light->GetColor())
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 640), L"Spot light properties"));

			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(
				WAF::Rect(20, 560, 260, 150)));

			// size
			mp_lSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 60, 15),
				L"Size: "));
			mp_tbSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(70, 0, 180, 40),
				WAF::Range(1, 100),
				m_light->GetSize() * 100.0f, 1u, 5u,
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
				WAF::Rect(60, 43, 100, 20), std::to_wstring(int(m_light->GetEmission())),
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
				m_light->GetBeamAngle() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				50u, false));
			mp_tbAngle->BindEventFunc(&SpotLightEditor::TBAngle_OnDrag, this);
			WriteAngle();
		}
		SpotLightEditor::~SpotLightEditor()
		{
			mp_gbProperties->Destroy();
			mp_pOthers->Destroy();
		}


		void SpotLightEditor::WriteSize()
		{
			wchar_t buffer[16];
			std::swprintf(buffer, 16, L"Size: %1.2f", m_light->GetSize());
			mp_lSize->SetCaption(buffer);
		}
		void SpotLightEditor::WriteAngle()
		{
			wchar_t buffer[32];
			std::swprintf(buffer, 32, L"Beam angle: %1.2f", m_light->GetBeamAngle());
			mp_lAngle->SetCaption(buffer);
		}

		void SpotLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_light->SetSize(mp_tbSize->GetPosition() / 100.0f);
			WriteSize();
		}
		void SpotLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				m_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				m_light->SetEmission(1.0f);
			}
		}
		void SpotLightEditor::TBAngle_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_light->SetBeamAngle(mp_tbAngle->GetPosition() / 100.0f);
			WriteAngle();
		}

		void SpotLightEditor::NotifyPosition(const Math::vec3f& position)
		{
			m_light->SetPosition(position);
		}
		void SpotLightEditor::NotifyDirection(const Math::vec3f& direction)
		{
			m_light->SetDirection(direction);
		}
		void SpotLightEditor::NotifyColor(const Graphics::Color& color)
		{
			m_light->SetColor(color);
		}



		// ~~~~~~~~ [CLASS] DirectLightEditor ~~~~~~~~
		DirectLightEditor::DirectLightEditor(
			WAF::Window* window, 
			const RZ::Handle<RZ::DirectLight>& light)
			: mp_window(window)
			, m_light(light)
			, m_direction_editor(
				mp_window,
				WAF::Point(20, 100),
				std::bind(&DirectLightEditor::NotifyDirection, this, std::placeholders::_1),
				m_light->GetDirection())
			, m_color_editor(
				mp_window,
				WAF::Point(20, 250),
				std::bind(&DirectLightEditor::NotifyColor, this, std::placeholders::_1),
				m_light->GetColor())
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 300), L"Direct light properties"));
			mp_pOthers = mp_window->CreateChild(WAF::ConStruct<WAF::Panel>(WAF::Rect(20, 430, 260, 100)));

			// size
			mp_lSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 15, 60, 20),
				L"Size: "));
			mp_tbSize = mp_pOthers->CreateChild(WAF::ConStruct<WAF::TrackBar>(
				WAF::Rect(65, 5, 185, 40),
				WAF::Range(1, 314),
				m_light->GetAngularSize() * 100.0f, 1u, 5u,
				WAF::TrackBar::Orientation::Horizontal,
				WAF::TrackBar::TickStyle::Both,
				10u, false));
			mp_tbSize->BindEventFunc(&DirectLightEditor::TBSize_OnDrag, this);
			WriteSize();

			// emission
			mp_lEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Label>(
				WAF::Rect(5, 57, 60, 20),
				L"Emission: "));
			mp_eEmission = mp_pOthers->CreateChild(WAF::ConStruct<WAF::Edit>(
				WAF::Rect(70, 55, 100, 20), std::to_wstring(int(m_light->GetEmission())),
				L"",
				WAF::Edit::TextAlignment::Left,
				WAF::Edit::LettersMode::All,
				false, true, false, false, false, 6));
			mp_eEmission->BindEventFunc(&DirectLightEditor::EditEmission_OnInput, this);
		}
		DirectLightEditor::~DirectLightEditor()
		{
			mp_gbProperties->Destroy();
			mp_pOthers->Destroy();
		}

		void DirectLightEditor::WriteSize()
		{
			wchar_t buffer[16];
			std::swprintf(buffer, 16, L"Size: %1.2f", m_light->GetAngularSize());
			mp_lSize->SetCaption(buffer);
		}
		void DirectLightEditor::NotifyDirection(const Math::vec3f& direction)
		{
			m_light->SetDirection(direction);
		}
		void DirectLightEditor::NotifyColor(const Graphics::Color& color)
		{
			m_light->SetColor(color);
		}

		void DirectLightEditor::TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event)
		{
			m_light->SetAngularSize(mp_tbSize->GetPosition() / 100.0f);
			WriteSize();
		}
		void DirectLightEditor::EditEmission_OnInput(WAF::Edit::Events::EventSetText& event)
		{
			try
			{
				m_light->SetEmission(std::stoi(mp_eEmission->GetText()));
			}
			catch (const std::invalid_argument&)
			{
				m_light->SetEmission(1.0f);
			}
		}



		// ~~~~~~~~ [CLASS] SphereEditor ~~~~~~~~
		SphereEditor::SphereEditor(
			WAF::Window* window, 
			const RZ::Handle<RZ::Sphere>& sphere)
			: mp_window(window)
			, m_sphere(sphere)
			, m_position_editor(
				mp_window,
				WAF::Point(20, 100),
				std::bind(&SphereEditor::NotifyPosition, this, std::placeholders::_1),
				sphere->GetTransformation().GetPosition())
			, m_rotation_editor(
				mp_window,
				WAF::Point(20, 230),
				std::bind(&SphereEditor::NotifyRotation, this, std::placeholders::_1),
				sphere->GetTransformation().GetRotation())
			, m_center_editor(
				mp_window,
				WAF::Point(20, 360),
				std::bind(&SphereEditor::NotifyCenter, this, std::placeholders::_1),
				sphere->GetTransformation().GetCenter())
			, m_scale_editor(
				mp_window,
				WAF::Point(20, 490),
				std::bind(&SphereEditor::NotifyScale, this, std::placeholders::_1),
				sphere->GetTransformation().GetScale())
			, m_material_editor(
				mp_window,
				sphere->GetMaterial(),
				WAF::Point(20, 620))
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 900), L"Sphere properties"));
		}
		SphereEditor::~SphereEditor()
		{
			mp_gbProperties->Destroy();
		}

		void SphereEditor::NotifyPosition(const Math::vec3f& position)
		{
			m_sphere->SetPosition(position);
		}
		void SphereEditor::NotifyRotation(const Math::vec3f& rotation)
		{
			m_sphere->SetRotation(rotation);
		}
		void SphereEditor::NotifyCenter(const Math::vec3f& center)
		{
			m_sphere->SetCenter(center);
		}
		void SphereEditor::NotifyScale(const Math::vec3f& scale)
		{
			m_sphere->SetScale(scale);
		}


		// ~~~~~~~~ [CLASS] MeshEditor ~~~~~~~~
		MeshEditor::MeshEditor(
			WAF::Window* window, 
			const RZ::Handle<RZ::Mesh>& mesh)
			: mp_window(window)
			, m_mesh(mesh)
			, m_position_editor(
				mp_window,
				WAF::Point(20, 100),
				std::bind(&MeshEditor::NotifyPosition, this, std::placeholders::_1),
				mesh->GetTransformation().GetPosition())
			, m_rotation_editor(
				mp_window,
				WAF::Point(20, 230),
				std::bind(&MeshEditor::NotifyRotation, this, std::placeholders::_1),
				mesh->GetTransformation().GetRotation())
			, m_center_editor(
				mp_window,
				WAF::Point(20, 360),
				std::bind(&MeshEditor::NotifyCenter, this, std::placeholders::_1),
				mesh->GetTransformation().GetCenter())
			, m_scale_editor(
				mp_window,
				WAF::Point(20, 490),
				std::bind(&MeshEditor::NotifyScale, this, std::placeholders::_1),
				mesh->GetTransformation().GetScale())
			, m_material_editor(
				mp_window, 
				mesh->GetMaterial(0u), 
				WAF::Point(20, 620))
		{
			mp_gbProperties = mp_window->CreateChild(WAF::ConStruct<WAF::GroupBox>(
				WAF::Rect(10, 80, 280, 900), L"Mesh properties"));
		}
		MeshEditor::~MeshEditor()
		{
			mp_gbProperties->Destroy();
		}

		void MeshEditor::NotifyPosition(const Math::vec3f& position)
		{
			m_mesh->SetPosition(position);
		}
		void MeshEditor::NotifyRotation(const Math::vec3f& rotation)
		{
			m_mesh->SetRotation(rotation);
		}
		void MeshEditor::NotifyCenter(const Math::vec3f& center)
		{
			m_mesh->SetCenter(center);
		}
		void MeshEditor::NotifyScale(const Math::vec3f& scale)
		{
			m_mesh->SetScale(scale);
		}
	}
}