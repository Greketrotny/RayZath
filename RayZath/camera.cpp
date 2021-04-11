#include "camera.h"

namespace RayZath
{
	Camera::Camera(
		Updatable* updatable,
		const ConStruct<Camera>& conStruct)
		: WorldObject(updatable, conStruct)
	{
		// [>] position and rotation
		SetPosition(conStruct.position);
		SetRotation(conStruct.rotation);


		// [>] resolution
		Resize(conStruct.width, conStruct.height);


		// [>] Sampling
		m_enabled = conStruct.enabled;
		m_samples_count = 0u;


		SetFov(conStruct.fov);
		SetFocalDistance(conStruct.focal_distance);
		SetAperture(conStruct.aperture);
	}
	Camera::~Camera()
	{
		if (mp_bitmap)
		{
			delete mp_bitmap;
			mp_bitmap = nullptr;
		}
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
	void Camera::Resize(const uint32_t& width, const uint32_t& height)
	{
		if (width == m_width && height == m_height)
			return;

		m_width = width;
		m_height = height;

		m_aspect_ratio = (float)m_width / (float)m_height;

		if (mp_bitmap) mp_bitmap->Resize(m_width, m_height);
		else mp_bitmap = new Graphics::Bitmap(m_width, m_height);

		GetStateRegister().RequestUpdate();
	}
	void Camera::SetPixel(const uint32_t& x, const uint32_t& y, const Graphics::Color& color)
	{
		mp_bitmap->SetPixel(std::min(x, m_width), std::min(y, m_height), color);
	}
	void Camera::LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle)
	{
		LookInDirection(point - m_position);
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

	uint32_t Camera::GetWidth() const
	{
		return m_width;
	}
	uint32_t Camera::GetHeight() const
	{
		return m_height;
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
	float Camera::GetFocalDistance() const
	{
		return m_focal_distance;
	}
	float Camera::GetAperture() const
	{
		return m_aperture;
	}
	const uint32_t& Camera::GetSamplesCount() const
	{
		return m_samples_count;
	}

	const Graphics::Bitmap& Camera::GetBitmap() const
	{
		return *mp_bitmap;
	}
}