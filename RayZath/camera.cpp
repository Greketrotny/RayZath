#include "camera.hpp"

#include "instance.hpp"
#include "material.hpp"

namespace RayZath::Engine
{
	Camera::Camera(
		Updatable* updatable,
		const ConStruct<Camera>& conStruct)
		: WorldObject(updatable, conStruct)
		, m_ray_count(0u)
	{
		position(conStruct.position);
		rotation(conStruct.rotation);

		resize(conStruct.resolution);

		m_enabled = conStruct.enabled;

		fov(conStruct.fov);
		nearFar(conStruct.near_far);
		focalDistance(conStruct.focal_distance);
		aperture(conStruct.aperture);
		exposureTime(conStruct.exposure_time);
		temporalBlend(conStruct.temporal_blend);
	}


	void Camera::enableRender()
	{
		m_enabled = true;
		stateRegister().RequestUpdate();
	}
	void Camera::disableRender()
	{
		m_enabled = false;
		stateRegister().RequestUpdate();
	}
	bool Camera::enabled() const
	{
		return m_enabled;
	}
	void Camera::resize(const Math::vec2ui32& resolution)
	{
		if (m_resolution == resolution)
			return;

		m_resolution = Math::vec2ui32(
			std::max(resolution.x, uint32_t(1)), 
			std::max(resolution.y, uint32_t(1)));
		m_aspect_ratio = float(m_resolution.x) / float(m_resolution.y);

		m_image_buffer.Resize(m_resolution.x, m_resolution.y);
		m_depth_buffer.Resize(m_resolution.x, m_resolution.y);

		m_focal_point = m_ray_cast_pixel = m_resolution / 2;		

		stateRegister().RequestUpdate();
	}
	void Camera::setPixel(const Math::vec2ui32& pixel, const Graphics::Color& color)
	{
		m_image_buffer.Value(
			std::min(pixel.x, m_resolution.x),
			std::min(pixel.y, m_resolution.y)) = color;
	}
	void Camera::lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		lookInDirection(point - m_position, angle);
	}
	void Camera::lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle)
	{
		const Math::vec3f dir = direction.Normalized();
		const float x_angle = asin(dir.y);
		const float y_angle = -atan2f(dir.x, dir.z);
		m_rotation = Math::vec3f(x_angle, y_angle, angle.value());
		m_coord_system.lookAt(m_rotation);
		stateRegister().RequestUpdate();
	}
	void Camera::focus(const Math::vec2ui32& pixel)
	{
		if (m_resolution.x == 0 || m_resolution.y == 0) return;

		m_focal_point = Math::vec2ui32(
			std::min(pixel.x, uint32_t(depthBuffer().GetWidth() - 1u)),
			std::min(pixel.y, uint32_t(depthBuffer().GetHeight() - 1u)));
		focalDistance(depthBuffer().Value(m_focal_point.x, m_focal_point.y));
	}

	void Camera::position(const Math::vec3f& position)
	{
		m_position = position;
		stateRegister().RequestUpdate();
	}
	void Camera::rotation(const Math::vec3f& rotation)
	{
		m_rotation = rotation;
		m_coord_system.lookAt(m_rotation);
		stateRegister().RequestUpdate();
	}
	const CoordSystem& Camera::coordSystem() const
	{
		return m_coord_system;
	}
	void Camera::fov(const Math::angle_radf& fov)
	{
		m_fov = fov;

		if (m_fov.value() < std::numeric_limits<float>::epsilon())
			m_fov.value() = std::numeric_limits<float>::epsilon();
		else if (m_fov.value() > Math::constants<float>::pi - std::numeric_limits<float>::epsilon())
			m_fov.value() = Math::constants<float>::pi - std::numeric_limits<float>::epsilon();

		stateRegister().RequestUpdate();
	}
	void Camera::nearFar(const Math::vec2f& near_far)
	{
		m_near_far = near_far;

		if (m_near_far.x < std::numeric_limits<float>::epsilon())
			m_near_far.x = std::numeric_limits<float>::epsilon();
		if (m_near_far.y < m_near_far.x + std::numeric_limits<float>::epsilon())
			m_near_far.y = m_near_far.x + std::numeric_limits<float>::epsilon();

		stateRegister().RequestUpdate();
	}
	void Camera::nearFar(const float& near, const float& far)
	{
		nearFar(Math::vec2f(near, far));
	}

	void Camera::focalDistance(float focal_distance)
	{
		m_focal_distance = focal_distance;
		if (m_focal_distance < std::numeric_limits<float>::epsilon())
			m_focal_distance = std::numeric_limits<float>::epsilon();

		stateRegister().RequestUpdate();
	}
	void Camera::aperture(float aperture)
	{
		m_aperture = aperture;
		if (m_aperture < std::numeric_limits<float>::epsilon())
			m_aperture = std::numeric_limits<float>::epsilon();
		stateRegister().RequestUpdate();
	}
	void Camera::exposureTime(float exposure_time)
	{
		m_exposure_time = exposure_time;
		if (m_exposure_time < std::numeric_limits<float>::epsilon())
			m_exposure_time = std::numeric_limits<float>::epsilon();
		stateRegister().RequestUpdate();
	}
	void Camera::temporalBlend(float temporal_blend)
	{
		m_temporal_blend = std::clamp(temporal_blend, 0.0f, 1.0f);
		stateRegister().RequestUpdate();
	}
	void Camera::rayCastPixel(Math::vec2ui32 pixel)
	{
		if (pixel.x >= m_resolution.x) pixel.x = m_resolution.x - 1;
		if (pixel.y >= m_resolution.y) pixel.y = m_resolution.y - 1;
		m_ray_cast_pixel = pixel;
		stateRegister().MakeModified();
	}

	uint32_t Camera::width() const
	{
		return m_resolution.x;
	}
	uint32_t Camera::height() const
	{
		return m_resolution.y;
	}
	const Math::vec2ui32& Camera::resolution() const
	{
		return m_resolution;
	}
	float Camera::aspectRatio() const
	{
		return m_aspect_ratio;
	}

	const Math::vec3f& Camera::position() const
	{
		return m_position;
	}
	const Math::vec3f& Camera::rotation() const
	{
		return m_rotation;
	}

	const Math::angle_radf& Camera::fov() const
	{
		return m_fov;
	}
	const Math::vec2f& Camera::nearFar() const
	{
		return m_near_far;
	}
	const float& Camera::nearDistance() const
	{
		return m_near_far.x;
	}
	const float& Camera::farDistance() const
	{
		return m_near_far.y;
	}

	float Camera::focalDistance() const
	{
		return m_focal_distance;
	}
	const Math::vec2ui32& Camera::focalPoint() const
	{
		return m_focal_point;
	}
	float Camera::aperture() const
	{
		return m_aperture;
	}
	float Camera::exposureTime() const
	{
		return m_exposure_time;
	}
	float Camera::temporalBlend() const
	{
		return m_temporal_blend;
	}
	Math::vec2ui32 Camera::getRayCastPixel() const
	{
		return m_ray_cast_pixel;
	}

	uint64_t Camera::rayCount() const
	{
		return m_ray_count;
	}
	void Camera::rayCount(const uint64_t ray_count)
	{
		m_ray_count = ray_count;
	}

	Graphics::Bitmap& Camera::imageBuffer()
	{
		return m_image_buffer;
	}
	const Graphics::Bitmap& Camera::imageBuffer() const
	{
		return m_image_buffer;
	}
	Graphics::Buffer2D<float>& Camera::depthBuffer() 
	{
		return m_depth_buffer;
	}
	const Graphics::Buffer2D<float>& Camera::depthBuffer() const
	{
		return m_depth_buffer;
	}
}
