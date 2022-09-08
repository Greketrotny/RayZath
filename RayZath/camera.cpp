#include "camera.hpp"

namespace RayZath::Engine
{
	Camera::Camera(
		Updatable* updatable,
		const ConStruct<Camera>& conStruct)
		: WorldObject(updatable, conStruct)
		, m_ray_count(0u)
	{
		SetPosition(conStruct.position);
		SetRotation(conStruct.rotation);

		Resize(conStruct.resolution);

		m_enabled = conStruct.enabled;

		SetFov(conStruct.fov);
		SetNearFar(conStruct.near_far);
		SetFocalDistance(conStruct.focal_distance);
		SetAperture(conStruct.aperture);
		SetExposureTime(conStruct.exposure_time);
		SetTemporalBlend(conStruct.temporal_blend);
	}


	void Camera::EnableRender()
	{
		m_enabled = true;
		GetStateRegister().RequestUpdate();
	}
	void Camera::DisableRender()
	{
		m_enabled = false;
		GetStateRegister().RequestUpdate();
	}
	bool Camera::Enabled() const
	{
		return m_enabled;
	}
	void Camera::Resize(const Math::vec2ui32& resolution)
	{
		if (m_resolution == resolution)
			return;

		m_resolution = Math::vec2ui32(
			std::max(resolution.x, uint32_t(1)), 
			std::max(resolution.y, uint32_t(1)));
		m_aspect_ratio = float(m_resolution.x) / float(m_resolution.y);

		m_image_buffer.Resize(m_resolution.x, m_resolution.y);
		m_depth_buffer.Resize(m_resolution.x, m_resolution.y);

		GetStateRegister().RequestUpdate();
	}
	void Camera::SetPixel(const Math::vec2ui32& pixel, const Graphics::Color& color)
	{
		m_image_buffer.Value(
			std::min(pixel.x, m_resolution.x),
			std::min(pixel.y, m_resolution.y)) = color;
	}
	void Camera::LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		LookInDirection(point - m_position, angle);
	}
	void Camera::LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle)
	{
		const Math::vec3f dir = direction.Normalized();
		const float x_angle = asin(dir.y);
		const float y_angle = -atan2f(dir.x, dir.z);
		m_rotation = Math::vec3f(x_angle, y_angle, angle.value());
		m_coord_system.LookAt(m_rotation);
		GetStateRegister().RequestUpdate();
	}
	void Camera::Focus(const Math::vec2ui32& pixel)
	{
		if (m_resolution.x == 0 || m_resolution.y == 0) return;

		m_focal_point = Math::vec2ui32(
			std::min(pixel.x, uint32_t(GetDepthBuffer().GetWidth() - 1u)),
			std::min(pixel.y, uint32_t(GetDepthBuffer().GetHeight() - 1u)));
		SetFocalDistance(GetDepthBuffer().Value(m_focal_point.x, m_focal_point.y));
	}

	void Camera::SetPosition(const Math::vec3f& position)
	{
		m_position = position;
		GetStateRegister().RequestUpdate();
	}
	void Camera::SetRotation(const Math::vec3f& rotation)
	{
		m_rotation = rotation;
		m_coord_system.LookAt(m_rotation);
		GetStateRegister().RequestUpdate();
	}
	const CoordSystem& Camera::GetCoordSystem() const
	{
		return m_coord_system;
	}
	void Camera::SetFov(const Math::angle_radf& fov)
	{
		m_fov = fov;

		if (m_fov.value() < std::numeric_limits<float>::epsilon())
			m_fov.value() = std::numeric_limits<float>::epsilon();
		else if (m_fov.value() > Math::constants<float>::pi - std::numeric_limits<float>::epsilon())
			m_fov.value() = Math::constants<float>::pi - std::numeric_limits<float>::epsilon();

		GetStateRegister().RequestUpdate();
	}
	void Camera::SetNearFar(const Math::vec2f& near_far)
	{
		m_near_far = near_far;

		if (m_near_far.x < std::numeric_limits<float>::epsilon())
			m_near_far.x = std::numeric_limits<float>::epsilon();
		if (m_near_far.y < m_near_far.x + std::numeric_limits<float>::epsilon())
			m_near_far.y = m_near_far.x + std::numeric_limits<float>::epsilon();

		GetStateRegister().RequestUpdate();
	}
	void Camera::SetNearFar(const float& near, const float& far)
	{
		SetNearFar(Math::vec2f(near, far));
	}

	void Camera::SetFocalDistance(float focal_distance)
	{
		m_focal_distance = focal_distance;
		if (m_focal_distance < std::numeric_limits<float>::epsilon())
			m_focal_distance = std::numeric_limits<float>::epsilon();

		GetStateRegister().RequestUpdate();
	}
	void Camera::SetAperture(float aperture)
	{
		m_aperture = aperture;
		if (m_aperture < std::numeric_limits<float>::epsilon())
			m_aperture = std::numeric_limits<float>::epsilon();
		GetStateRegister().RequestUpdate();
	}
	void Camera::SetExposureTime(float exposure_time)
	{
		m_exposure_time = exposure_time;
		if (m_exposure_time < std::numeric_limits<float>::epsilon())
			m_exposure_time = std::numeric_limits<float>::epsilon();
		GetStateRegister().RequestUpdate();
	}
	void Camera::SetTemporalBlend(float temporal_blend)
	{
		m_temporal_blend = std::clamp(temporal_blend, 0.0f, 1.0f);
		GetStateRegister().RequestUpdate();
	}

	uint32_t Camera::GetWidth() const
	{
		return m_resolution.x;
	}
	uint32_t Camera::GetHeight() const
	{
		return m_resolution.y;
	}
	const Math::vec2ui32& Camera::GetResolution() const
	{
		return m_resolution;
	}
	float Camera::GetAspectRatio() const
	{
		return m_aspect_ratio;
	}

	const Math::vec3f& Camera::GetPosition() const
	{
		return m_position;
	}
	const Math::vec3f& Camera::GetRotation() const
	{
		return m_rotation;
	}

	const Math::angle_radf& Camera::GetFov() const
	{
		return m_fov;
	}
	const Math::vec2f& Camera::GetNearFar() const
	{
		return m_near_far;
	}
	const float& Camera::GetNearDistance() const
	{
		return m_near_far.x;
	}
	const float& Camera::GetFarDistance() const
	{
		return m_near_far.y;
	}

	float Camera::GetFocalDistance() const
	{
		return m_focal_distance;
	}
	const Math::vec2ui32& Camera::GetFocalPoint() const
	{
		return m_focal_point;
	}
	float Camera::GetAperture() const
	{
		return m_aperture;
	}
	float Camera::GetExposureTime() const
	{
		return m_exposure_time;
	}
	float Camera::GetTemporalBlend() const
	{
		return m_temporal_blend;
	}
	uint64_t Camera::GetRayCount() const
	{
		return m_ray_count;
	}

	Graphics::Bitmap& Camera::GetImageBuffer()
	{
		return m_image_buffer;
	}
	const Graphics::Bitmap& Camera::GetImageBuffer() const
	{
		return m_image_buffer;
	}
	Graphics::Buffer2D<float>& Camera::GetDepthBuffer() 
	{
		return m_depth_buffer;
	}
	const Graphics::Buffer2D<float>& Camera::GetDepthBuffer() const
	{
		return m_depth_buffer;
	}
}