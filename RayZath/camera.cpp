#include "camera.h"

namespace RayZath
{
	Camera::Camera(const ConStruct<Camera>& conStruct, Updatable* updatable)
		: WorldObject(conStruct, updatable)
	{
		// [>] position and rotation
		SetPosition(conStruct.position);
		SetRotation(conStruct.rotation);


		// [>] resolution
		m_max_width = conStruct.max_width;
		if (conStruct.width > m_max_width) m_width = m_max_width;
		else m_width = conStruct.width;

		m_max_height = conStruct.max_height;
		if (conStruct.height > m_max_height) m_height = m_max_height;
		else m_height = conStruct.height;

		m_aspect_ratio = (float)m_width / (float)m_height;


		// [>] Sampling
		m_enabled = conStruct.enabled;
		m_samples_count = 0u;


		SetFov(conStruct.fov);
		SetFocalDistance(conStruct.focal_distance);
		SetAperture(conStruct.aperture);


		// [>] Bitmap
		mp_bitmap = new Graphics::Bitmap(m_width, m_height);
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
	}
	void Camera::DisableRender()
	{
		m_enabled = false;
	}
	bool Camera::Enabled() const
	{
		return m_enabled;
	}
	void Camera::Resize(const size_t& width, const size_t& height)
	{
		if (width == m_width && height == m_height)
			return;

		m_width = std::min(width, m_max_width);
		m_height = std::min(height, m_max_height);

		m_aspect_ratio = (float)m_width / (float)m_height;
		if (mp_bitmap) mp_bitmap->Resize(m_width, m_height);
	}
	void Camera::SetPixel(const size_t& x, const size_t& y, const Graphics::Color& color)
	{
		mp_bitmap->SetPixel(x, y, color);
	}

	void Camera::SetPosition(const Math::vec3<float>& newPosition)
	{
		m_position = newPosition;
		RequestUpdate();
	}
	void Camera::SetRotation(const Math::vec3<float>& newRotation)
	{
		m_rotation = newRotation;
		//m_rotation.x = fmod(m_rotation.x, Math::constants<float>::Tau);
		//if (m_rotation.x < 0.0f) m_rotation.x += Math::constants<float>::Tau;
		//m_rotation.y = fmod(m_rotation.y, Math::constants<float>::Tau);
		//if (m_rotation.y < 0.0f) m_rotation.y += Math::constants<float>::Tau;
		//m_rotation.z = fmod(m_rotation.z, Math::constants<float>::Tau);
		//if (m_rotation.z < 0.0f) m_rotation.z += Math::constants<float>::Tau;
		RequestUpdate();
	}
	void Camera::SetFov(const Math::angle<Math::rad, float>& fov)
	{
		m_fov = fov;

		if (m_fov.value() < std::numeric_limits<float>::epsilon())
			m_fov.value() = std::numeric_limits<float>::epsilon();
		else if (m_fov.value() > Math::constants<float>::Pi - std::numeric_limits<float>::epsilon())
			m_fov.value() = Math::constants<float>::Pi - std::numeric_limits<float>::epsilon();

		RequestUpdate();
	}
	void Camera::SetFocalDistance(float focal_distance)
	{
		m_focal_distance = focal_distance;
		if (m_focal_distance < std::numeric_limits<float>::epsilon())
			m_focal_distance = std::numeric_limits<float>::epsilon();

		RequestUpdate();
	}
	void Camera::SetAperture(float aperture)
	{
		m_aperture = aperture;
		if (m_aperture < std::numeric_limits<float>::epsilon())
			m_aperture = std::numeric_limits<float>::epsilon();

		RequestUpdate();
	}

	size_t Camera::GetWidth() const
	{
		return m_width;
	}
	size_t Camera::GetMaxWidth() const
	{
		return m_max_width;
	}
	size_t Camera::GetHeight() const
	{
		return m_height;
	}
	size_t Camera::GetMaxHeight() const
	{
		return m_max_height;
	}
	float Camera::GetAspectRatio() const
	{
		return m_aspect_ratio;
	}

	const Math::vec3<float>& Camera::GetPosition() const
	{
		return m_position;
	}
	const Math::vec3<float>& Camera::GetRotation() const
	{
		return m_rotation;
	}
	const Math::angle<Math::rad, float>& Camera::GetFov() const
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
	const Graphics::Bitmap& Camera::GetBitmap() const
	{
		return *mp_bitmap;
	}
}