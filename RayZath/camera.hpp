#ifndef CAMERA_H
#define CAMERA_H

#include "world_object.hpp"
#include "render_parts.hpp"

#include "vec3.h"
#include "vec2.h"
#include "angle.h"

#include "bitmap.h"
#include "color.h"
#include "constants.h"

namespace RayZath::Cuda
{
	class EngineCore;
}

namespace RayZath::Engine
{
	class Camera;
	template<> struct ConStruct<Camera>;

	class Instance;
	struct Material;

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
		Math::vec2ui32 m_focal_point;
		float m_aperture;
		float m_exposure_time;

		bool m_enabled;

		uint64_t m_ray_count;
		float m_temporal_blend;

		Graphics::Bitmap m_image_buffer;
		Graphics::Buffer2D<float> m_depth_buffer;

		Math::vec2ui32 m_ray_cast_pixel;
	public:
		Handle<Instance> m_raycasted_instance;
		Handle<Material> m_raycasted_material;


		Camera(const Camera&) = delete;
		Camera(Camera&&) = delete;
		Camera(
			Updatable* updatable, 
			const ConStruct<Camera>& conStruct);


		Camera& operator=(const Camera&) = delete;
		Camera& operator=(Camera&&) = delete;


		void enableRender();
		void disableRender();
		bool enabled() const;
		void resize(const Math::vec2ui32& resolution);
		void setPixel(const Math::vec2ui32& pixel, const Graphics::Color& color);
		void lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);
		void focus(const Math::vec2ui32& pixel);

		void position(const Math::vec3f& position);
		void rotation(const Math::vec3f& rotation);

		void fov(const Math::angle_radf& fov);
		void nearFar(const Math::vec2f& near_far);
		void nearFar(const float& near, const float& far);

		void focalDistance(float focal_distance);
		void aperture(float aperture);
		void exposureTime(float exposure_time);
		void temporalBlend(float temporal_blend);
		void rayCastPixel(const Math::vec2ui32 pixel);


		uint32_t width() const;
		uint32_t height() const;
		const Math::vec2ui32& resolution() const;
		float aspectRatio() const;

		const Math::vec3f& position() const;
		const Math::vec3f& rotation() const;
		const CoordSystem& coordSystem() const;

		const Math::angle_radf& fov() const;
		const Math::vec2f& nearFar() const;
		const float& nearDistance() const;
		const float& farDistance() const;

		float focalDistance() const;
		const Math::vec2ui32& focalPoint() const;
		float aperture() const;
		float exposureTime() const;
		float temporalBlend() const;
		Math::vec2ui32 getRayCastPixel() const;
		uint64_t rayCount() const;

		Graphics::Bitmap& imageBuffer();
		const Graphics::Bitmap& imageBuffer() const;
		Graphics::Buffer2D<float>& depthBuffer();
		const Graphics::Buffer2D<float>& depthBuffer() const;

		friend class RayZath::Cuda::EngineCore;
	};


	template<> struct ConStruct<Camera> : public ConStruct<WorldObject>
	{
		Math::vec3f position = Math::vec3f(0.0f, 0.0f, -10.0f);
		Math::vec3f rotation = Math::vec3f(0.0f, 0.0f, 0.0f);
		Math::vec2ui32 resolution = Math::vec2ui32(1280u, 720u);
		Math::angle_radf fov = Math::constants<float>::pi / 2.0f;
		Math::vec2f near_far = Math::vec2f(1.0e-2f, 1.0e+3f);
		float focal_distance = 10.0f;
		float aperture = 0.02f;
		float exposure_time = 1.0f / 60.0f;
		float temporal_blend = 0.75f;
		bool enabled = true;

		ConStruct(
			const std::string& name = "name",
			const Math::vec3f& position = Math::vec3f(0.0f, 0.0f, -10.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec2ui32& resolution = Math::vec2ui32(1280u, 720u),
			const Math::angle_radf& fov = Math::constants<float>::pi / 2.0f,
			const Math::vec2f& near_far = Math::vec2f(1.0e-2f, 1.0e+3f),
			float focal_distance = 10.0f,
			float aperture = 0.02f,
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
		ConStruct(const Handle<Camera>& camera)
		{
			if (!camera) return;

			name = camera->name();
			position = camera->position();
			rotation = camera->rotation();
			resolution = camera->resolution();
			fov = camera->fov();
			near_far = camera->nearFar();
			focal_distance = camera->focalDistance();
			aperture = camera->aperture();
			exposure_time = camera->exposureTime();
			temporal_blend = camera->temporalBlend();
			enabled = camera->enabled();
		}
	};
}

#endif // !CAMERA_H