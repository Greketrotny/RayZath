#ifndef PROPERTIES_EDITORS_H
#define PROPERTIES_EDITORS_H

#include "winapi_framework.h"
namespace WAF = WinapiFramework;

#include "rayzath.h"
namespace RZ = RayZath;

namespace Tester
{
	namespace UI
	{
		class PropsEditor
		{
		public:
			PropsEditor() = default;
			virtual ~PropsEditor() = default;

		public:
			virtual void UpdateState() {};
		};

		struct PositionEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::RenderObject* mp_object;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pPosition;
			WAF::Label* mp_lPosition;
			WAF::Label* mp_lPosX;
			WAF::TrackBar* mp_tbPosX;
			WAF::Label* mp_lPosY;
			WAF::TrackBar* mp_tbPosY;
			WAF::Label* mp_lPosZ;
			WAF::TrackBar* mp_tbPosZ;


		public:
			PositionEditor(WAF::Window* window, RZ::RenderObject* object,
				const WAF::Point& position);
			~PositionEditor();

			void WritePosition(const Math::vec3<float>& pos);

			void TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct RotationEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::RenderObject* mp_object;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pRotation;
			WAF::Label* mp_lRotation;
			WAF::Label* mp_lRotX;
			WAF::TrackBar* mp_tbRotX;
			WAF::Label* mp_lRotY;
			WAF::TrackBar* mp_tbRotY;
			WAF::Label* mp_lRotZ;
			WAF::TrackBar* mp_tbRotZ;


		public:
			RotationEditor(WAF::Window* window, RZ::RenderObject* object,
				const WAF::Point& position);
			~RotationEditor();

			void WriteRotation(const Math::vec3<float>& pos);

			void TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct ScaleEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::RenderObject* mp_object;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pScale;
			WAF::Label* mp_lScale;
			WAF::Label* mp_lScaleX;
			WAF::TrackBar* mp_tbScaleX;
			WAF::Label* mp_lScaleY;
			WAF::TrackBar* mp_tbScaleY;
			WAF::Label* mp_lScaleZ;
			WAF::TrackBar* mp_tbScaleZ;


		public:
			ScaleEditor(WAF::Window* window, RZ::RenderObject* object,
				const WAF::Point& position);
			~ScaleEditor();

			void WriteScale(const Math::vec3<float>& pos);

			void TBScaleX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBScaleY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBScaleZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};

		class CameraPropsEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Camera* mp_camera;

			// ~~~~ editors layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			// position
			WAF::Panel* mp_pPosition;
			WAF::Label* mp_lPosition;
			WAF::Label* mp_lPosX;
			WAF::TrackBar* mp_tbPosX;
			WAF::Label* mp_lPosY;
			WAF::TrackBar* mp_tbPosY;
			WAF::Label* mp_lPosZ;
			WAF::TrackBar* mp_tbPosZ;

			// rotation
			WAF::Panel* mp_pRotation;
			WAF::Label* mp_lRotation;
			WAF::Label* mp_lRotX;
			WAF::TrackBar* mp_tbRotX;
			WAF::Label* mp_lRotY;
			WAF::TrackBar* mp_tbRotY;
			WAF::Label* mp_lRotZ;
			WAF::TrackBar* mp_tbRotZ;

			// fov
			WAF::Panel* mp_pOthers;
			WAF::Label* mp_lFov;
			WAF::TrackBar* mp_tbFov;
			WAF::Label* mp_lFocalDistance;
			WAF::TrackBar* mp_tbFocalDistance;
			WAF::Label* mp_lAperature;
			WAF::TrackBar* mp_tbAperature;

		public:
			CameraPropsEditor(WAF::Window* window, RZ::Camera* camera);
			~CameraPropsEditor();


		public:
			void UpdateState() override;

			void WritePosition(const Math::vec3<float>& pos);
			void WriteRotation(const Math::vec3<float>& rot);

			// ~~~~ event handlers ~~~~
		public:
			void TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);

			void TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);

			void TBFov_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBFocalDistance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBAperature_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};

		class PointLightEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::PointLight* mp_light;

			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			// position
			WAF::Panel* mp_pPosition;
			WAF::Label* mp_lPosition;
			WAF::Label* mp_lPosX;
			WAF::TrackBar* mp_tbPosX;
			WAF::Label* mp_lPosY;
			WAF::TrackBar* mp_tbPosY;
			WAF::Label* mp_lPosZ;
			WAF::TrackBar* mp_tbPosZ;

			WAF::Panel* mp_pOthers;

			// size
			WAF::Label* mp_lSize;
			WAF::TrackBar* mp_tbSize;

			// emission
			WAF::Label* mp_lEmission;
			WAF::Edit* mp_eEmission;

		public:
			PointLightEditor(WAF::Window* window, RZ::PointLight* light);
			~PointLightEditor();

			void WritePosition(const Math::vec3<float>& pos);

			// ~~~~ event handlers ~~~~
			void TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);

			void TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);

			void EditEmission_OnInput(WAF::Edit::Events::EventSetText& event);
		};

		class SphereEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Sphere* mp_sphere;

			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_position_editor;
			RotationEditor m_rotation_editor;
			ScaleEditor m_scale_editor;


		public:
			SphereEditor(WAF::Window* window, RZ::Sphere* sphere);
			~SphereEditor();
		};
		//class MeshEditor : public PropsEditor
		//{
		//private:
		//	WAF::Window* mp_window;
		//	RZ::Mesh* mp_mesh;

		//	// ~~~~ editor layout ~~~~
		//	WAF::GroupBox* mp_gbProperties;

		//	PositionEditor m_position_editor;
		//	RotationEditor m_rotation_editor;
		//	ScaleEditor m_scale_editor;


		//public:
		//	MeshEditor(WAF::Window* window, RZ::Mesh* mesh);
		//	~MeshEditor();
		//};
	}
}

#endif // !PROPERTIES_EDITOR_H