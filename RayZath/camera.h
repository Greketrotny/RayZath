#ifndef CAMERA_H
#define CAMERA_H

#include "world_object.h"
#include "render_parts.h"

#include "vec3.h"
#include "vec2.h"
#include "angle.h"

#include "bitmap.h"
#include "color.h"
#include "constants.h"

namespace RayZath
{
	class Camera;
	template<> struct ConStruct<Camera>;

	namespace CudaEngine
	{
		class CudaEngineCore;
	}

	class Camera : public WorldObject
	{
	private:
		Math::vec3f m_position;
		Math::vec3f m_rotation;
		CoordSystem m_coord_system;

		Math::vec2ui32 m_resolution;
		float m_aspect_ratio;

		Math::angle<Math::angle_unit::rad, float> m_fov;
		Math::vec2f m_near_far;
		float m_focal_distance;
		float m_aperture;
		float m_exposure_time;

		bool m_enabled;

		uint32_t m_samples_count;
		float m_temporal_blend;

		Graphics::Bitmap m_image_buffer;
		Graphics::Buffer2D<float> m_depth_buffer;


	public:
		Camera(const Camera&) = delete;
		Camera(Camera&&) = delete;
		Camera(
			Updatable* updatable, 
			const ConStruct<Camera>& conStruct);


	public:
		Camera& operator=(const Camera&) = delete;
		Camera& operator=(Camera&&) = delete;


	public:
		void EnableRender();
		void DisableRender();
		bool Enabled() const;
		void Resize(const Math::vec2ui32& resolution);
		void SetPixel(const Math::vec2ui32& pixel, const Graphics::Color& color);
		void LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);
		void Focus(const Math::vec2ui32& pixel);

		void SetPosition(const Math::vec3f& position);
		void SetRotation(const Math::vec3f& rotation);

		void SetFov(const Math::angle_radf& fov);
		void SetNearFar(const Math::vec2f& near_far);
		void SetNearFar(const float& near, const float& far);

		void SetFocalDistance(float focal_distance);
		void SetAperture(float aperture);
		void SetExposureTime(float exposure_time);
		void SetTemporalBlend(float temporal_blend);


		uint32_t GetWidth() const;
		uint32_t GetHeight() const;
		const Math::vec2ui32& GetResolution() const;
		float GetAspectRatio() const;

		const Math::vec3f& GetPosition() const;
		const Math::vec3f& GetRotation() const;
		const CoordSystem& GetCoordSystem() const;

		const Math::angle_radf& GetFov() const;
		const Math::vec2f& GetNearFar() const;
		const float& GetNearDistance() const;
		const float& GetFarDistance() const;

		float GetFocalDistance() const;
		float GetAperture() const;
		float GetExposureTime() const;
		float GetTemporalBlend() const;
		const uint32_t& GetSamplesCount() const;

		const Graphics::Bitmap& GetImageBuffer() const;
		const Graphics::Buffer2D<float>& GetDepthBuffer() const;

		friend class CudaEngine::CudaEngineCore;
	};


	template<> struct ConStruct<Camera> : public ConStruct<WorldObject>
	{
		Math::vec3f position;
		Math::vec3f rotation;
		Math::vec2ui32 resolution;
		Math::angle_radf fov;
		Math::vec2f near_far;
		float focal_distance;
		float aperture;
		float exposure_time;
		float temporal_blend;
		bool enabled;

		ConStruct(
			const std::string& name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, -10.0f, 0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec2ui32& resolution = Math::vec2ui32(1280u, 720u),
			const Math::angle_radf& fov = Math::constants<float>::pi / 2.0f,
			const Math::vec2f& near_far = Math::vec2f(1.0e-2f, 1.0e+3f),
			float focal_distance = 10.0f,
			float aperture = 0.5f,
			float exposure_time = 1.0f / 60.0f,
			float temporal_blend = 0.75f,
			bool enabled = true)
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, resolution(resolution)
			, fov(fov)
			, near_far(near_far)
			, focal_distance(focal_distance)
			, aperture(aperture)
			, exposure_time(exposure_time)
			, temporal_blend(temporal_blend)
			, enabled(enabled)
		{}
	};
}

#endif // !CAMERA_H