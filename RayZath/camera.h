#ifndef CAMERA_H
#define CAMERA_H

#include "world_object.h"

#include "vec3.h"
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
		class Engine;
	}

	class Camera : public WorldObject
	{
	public:
		enum class Projection
		{
			Perspective,
			Orthographic,
			Spherical
		};
	private:
		Math::vec3<float> m_position;
		Math::vec3<float> m_rotation;

		uint32_t m_width, m_height;
		float m_aspect_ratio;

		Math::angle<Math::angle_unit::rad, float> m_fov;
		Projection m_projection;
		float m_focal_distance;
		float m_aperture;

		bool m_enabled;

		uint32_t m_samples_count;

		Graphics::Bitmap* mp_bitmap = nullptr;


	public:
		Camera(const Camera&) = delete;
		Camera(Camera&&) = delete;
		Camera(
			Updatable* updatable, 
			const ConStruct<Camera>& conStruct);
		~Camera();


	public:
		Camera& operator=(const Camera&) = delete;
		Camera& operator=(Camera&&) = delete;


	public:
		void EnableRender();
		void DisableRender();
		bool Enabled() const;
		void Resize(const uint32_t& width, const uint32_t& height);
		void SetPixel(const uint32_t& x, const uint32_t& y, const Graphics::Color& color);

		void SetPosition(const Math::vec3<float>& position);
		void SetRotation(const Math::vec3<float>& rotation);
		void SetFov(const Math::angle_radf& fov);
		void SetProjection(Projection projection);

		void SetFocalDistance(float focal_distance);
		void SetAperture(float aperture);


		uint32_t GetWidth() const;
		uint32_t GetHeight() const;
		float GetAspectRatio() const;

		const Math::vec3<float>& GetPosition() const;
		const Math::vec3<float>& GetRotation() const;
		const Math::angle_radf& GetFov() const;
		Projection GetProjection() const;
		float GetFocalDistance() const;
		float GetAperture() const;
		const uint32_t& GetSamplesCount() const;

		const Graphics::Bitmap& GetBitmap() const;

		friend class CudaEngine::Engine;
	};


	template<> struct ConStruct<Camera> : public ConStruct<WorldObject>
	{
		Math::vec3<float> position;
		Math::vec3<float> rotation;
		uint32_t width, height;
		Math::angle_radf fov;
		float focal_distance;
		float aperture;
		bool enabled;

		ConStruct(
			const std::wstring& name = L"name",
			const Math::vec3<float>& position = Math::vec3<float>(0.0f, -10.0f, 0.0f),
			const Math::vec3<float>& rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const uint32_t& width = 800u, const uint32_t& height = 600u,
			const Math::angle_radf& fov = Math::constants<float>::pi / 2.0f,
			float focal_distance = 10.0f,
			float aperture = 0.5f,
			bool enabled = true)
			: ConStruct<WorldObject>(name)
			, position(position)
			, rotation(rotation)
			, width(width)
			, height(height)
			, fov(fov)
			, focal_distance(focal_distance)
			, aperture(aperture)
			, enabled(enabled)
		{}
	};
}

#endif // !CAMERA_H