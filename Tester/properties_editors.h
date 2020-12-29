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
			std::function<void(const Math::vec3f&)> m_notify_function;
			Math::vec3f m_position;

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
			PositionEditor(
				WAF::Window* window,
				const WAF::Point& position,
				const std::function<void(const Math::vec3f&)> function,
				const Math::vec3f& initial_position);
			~PositionEditor();


			void SetPosition(const Math::vec3f& position);
		private:
			void WritePosition();
			void Notify();

			void TBPositionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBPositionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct RotationEditor
		{
		private:
			std::function<void(const Math::vec3f&)> m_notify_function;
			Math::vec3f m_rotation;

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
			RotationEditor(
				WAF::Window* window, 
				const WAF::Point& position,
				const std::function<void(const Math::vec3f&)> function,
				const Math::vec3f& initial_rotation);
			~RotationEditor();

			void SetRotation(const Math::vec3f& rotation);
		private:
			void WriteRotation();
			void Notify();

			void TBRotationX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBRotationZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct ScaleEditor
		{
		private:
			std::function<void(const Math::vec3f&)> m_notify_function;
			Math::vec3f m_scale;

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
			ScaleEditor(
				WAF::Window* window,
				const WAF::Point& position,
				const std::function<void(const Math::vec3f&)> function,
				const Math::vec3f& initial_scale);
			~ScaleEditor();



		private:
			void WriteScale();
			void Notify();

			void TBScaleX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBScaleY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBScaleZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct CenterEditor
		{
		private:
			std::function<void(const Math::vec3f&)> m_notify_function;
			Math::vec3f m_center;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pCenter;
			WAF::Label* mp_lCenter;
			WAF::Label* mp_lCenterX;
			WAF::TrackBar* mp_tbCenterX;
			WAF::Label* mp_lCenterY;
			WAF::TrackBar* mp_tbCenterY;
			WAF::Label* mp_lCenterZ;
			WAF::TrackBar* mp_tbCenterZ;


		public:
			CenterEditor(
				WAF::Window* window,
				const WAF::Point& position,
				const std::function<void(const Math::vec3f&)> function,
				const Math::vec3f& initial_center);
			~CenterEditor();



		private:
			void WriteCenter();
			void Notify();

			void TBCenterX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBCenterY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBCenterZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct DirectionEditor
		{
		private:
			std::function<void(const Math::vec3f&)> m_notify_function;
			Math::vec3f m_direction;
			float m_phi = 0.0f;
			float m_theta = -Math::constants<float>::half_pi;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pDirection;
			WAF::Label* mp_lDirection;
			WAF::Button* mp_bMode;
			enum class Mode
			{
				ByAxes,
				ByAngles
			} m_mode;

			// by axes
			WAF::Label* mp_lDirX;
			WAF::TrackBar* mp_tbDirX;
			WAF::Label* mp_lDirY;
			WAF::TrackBar* mp_tbDirY;
			WAF::Label* mp_lDirZ;
			WAF::TrackBar* mp_tbDirZ;
			// by angles
			WAF::Label* mp_lPhi;
			WAF::TrackBar* mp_tbPhi;
			WAF::Label* mp_lTheta;
			WAF::TrackBar* mp_tbTheta;


		public:
			DirectionEditor(
				WAF::Window* window,
				const WAF::Point& position,
				const std::function<void(const Math::vec3f&)> function,
				const Math::vec3f& initial_direction);
			~DirectionEditor();



		private:
			void Notify();
			void UpdateState();
			void WriteDirection();

			// ~~~~ event handlers ~~~~
			// button switch
			void BMode_OnClick(WAF::Button::Events::EventClick& event);
			// direction
			// by axes
			void TBDirectionX_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBDirectionY_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBDirectionZ_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			// by angles
			void TBDirectionPhi_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBDirectionTheta_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		struct MaterialEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::Material> m_material;

			// ~~~~ editor layout ~~~~
			WAF::Panel* mp_pMaterial;
			WAF::Label* mp_lMaterial;

			WAF::Label* mp_lReflectance;
			WAF::TrackBar* mp_tbReflectance;
			WAF::Label* mp_lGlossiness;
			WAF::TrackBar* mp_tbGlossiness;

			WAF::Label* mp_lTransmittance;
			WAF::TrackBar* mp_tbTransmittance;
			WAF::Label* mp_lIOR;
			WAF::TrackBar* mp_tbIOR;

			WAF::Label* mp_lScattering;
			WAF::TrackBar* mp_tbScattering;
			WAF::Label* mp_lEmission;
			WAF::Edit* mp_eEmission;


		public:
			MaterialEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::Material>& object,
				const WAF::Point& position);
			~MaterialEditor();


		public:
			void WriteMaterialProps();


			// ~~~~ event handlers ~~~~
		public:
			void TBReflectance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBGlossiness_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBTransmittance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBIOR_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBScattering_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void EEmission_OnEdit(WAF::Edit::Events::EventSetText& event);
		};

		class CameraPropsEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::Camera> m_camera;

			// ~~~~ editors layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_pos_editor;
			RotationEditor m_rot_editor;

			WAF::Panel* mp_pOthers;
			WAF::Label* mp_lFov;
			WAF::TrackBar* mp_tbFov;
			WAF::Label* mp_lFocalDistance;
			WAF::TrackBar* mp_tbFocalDistance;
			WAF::Label* mp_lAperature;
			WAF::TrackBar* mp_tbAperature;

		public:
			CameraPropsEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::Camera>& camera);
			~CameraPropsEditor();


		public:
			void UpdateState() override;

			void NotifyPosition(const Math::vec3f& position);
			void NotifyRotation(const Math::vec3f& rotation);

			// ~~~~ event handlers ~~~~
		public:
			void TBFov_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBFocalDistance_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void TBAperature_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};

		class PointLightEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::PointLight> m_light;


			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_position_editor;

			WAF::Panel* mp_pOthers;

			// size
			WAF::Label* mp_lSize;
			WAF::TrackBar* mp_tbSize;

			// emission
			WAF::Label* mp_lEmission;
			WAF::Edit* mp_eEmission;

		public:
			PointLightEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::PointLight>& light);
			~PointLightEditor();

			void NotifyPosition(const Math::vec3f& position);

			// ~~~~ event handlers ~~~~
			void TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void EditEmission_OnInput(WAF::Edit::Events::EventSetText& event);
		};
		class SpotLightEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::SpotLight> m_light;


			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_position_editor;
			DirectionEditor m_direction_editor;


			WAF::Panel* mp_pOthers;

			// size
			WAF::Label* mp_lSize;
			WAF::TrackBar* mp_tbSize;

			// emission
			WAF::Label* mp_lEmission;
			WAF::Edit* mp_eEmission;

			// beam angle
			WAF::Label* mp_lAngle;
			WAF::TrackBar* mp_tbAngle;


		public:
			SpotLightEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::SpotLight>& light);
			~SpotLightEditor();

			void WriteSize();
			void WriteAngle();

			void NotifyPosition(const Math::vec3f& position);
			void NotifyDirection(const Math::vec3f& direction);

			// ~~~~ event handlers ~~~~
			void TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void EditEmission_OnInput(WAF::Edit::Events::EventSetText& event);
			void TBAngle_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
		};
		class DirectLightEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::DirectLight> m_light;

			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			DirectionEditor m_direction_editor;

			// size
			WAF::Label* mp_lSize;
			WAF::TrackBar* mp_tbSize;
			// emission
			WAF::Label* mp_lEmission;
			WAF::Edit* mp_eEmission;


		public:
			DirectLightEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::DirectLight>& light);
			~DirectLightEditor();

			void WriteSize();
			void NotifyDirection(const Math::vec3f& direction);

			// ~~~~ event handlers ~~~~
			void TBSize_OnDrag(WAF::TrackBar::Events::EventDragThumb& event);
			void EditEmission_OnInput(WAF::Edit::Events::EventSetText& event);
		};

		class SphereEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::Sphere> m_sphere;

			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_position_editor;
			RotationEditor m_rotation_editor;
			CenterEditor m_center_editor;
			ScaleEditor m_scale_editor;
			MaterialEditor m_material_editor;


		public:
			SphereEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::Sphere>& sphere);
			~SphereEditor();


		public:
			void NotifyPosition(const Math::vec3f& position);
			void NotifyRotation(const Math::vec3f& rotation);
			void NotifyCenter(const Math::vec3f& center);
			void NotifyScale(const Math::vec3f& scale);
		};
		class MeshEditor : public PropsEditor
		{
		private:
			WAF::Window* mp_window;
			RZ::Handle<RZ::Mesh> m_mesh;

			// ~~~~ editor layout ~~~~
			WAF::GroupBox* mp_gbProperties;

			PositionEditor m_position_editor;
			RotationEditor m_rotation_editor;
			CenterEditor m_center_editor;
			ScaleEditor m_scale_editor;
			MaterialEditor m_material_editor;


		public:
			MeshEditor(
				WAF::Window* window, 
				const RZ::Handle<RZ::Mesh>& mesh);
			~MeshEditor();


		public:
			void NotifyPosition(const Math::vec3f& position);
			void NotifyRotation(const Math::vec3f& rotation);
			void NotifyCenter(const Math::vec3f& center);
			void NotifyScale(const Math::vec3f& scale);
		};
	}
}

#endif // !PROPERTIES_EDITOR_H
// 491 lines