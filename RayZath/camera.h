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
	template <typename T> struct ObjectContainer;

	class Camera;
	template<> struct ConStruct<Camera>;

	class Camera : public WorldObject
	{
	private:
		Math::vec3<float> m_position;
		Math::vec3<float> m_rotation;

		size_t m_width, m_height;
		float m_aspect_ratio;

		Math::angle<Math::AngleUnit::rad, float> m_fov;
		float m_focal_distance;
		float m_aperture;

		bool m_enabled;
	public:
		size_t m_samples_count;	// little leak for sample update
	private:

		Graphics::Bitmap* mp_bitmap = nullptr;


	public:
		Camera(const Camera&) = delete;
		Camera(Camera&&) = delete;
		Camera(const size_t& id, Updatable* updatable, const ConStruct<Camera>& conStruct);
		~Camera();


	public:
		Camera& operator=(const Camera&) = delete;
		Camera& operator=(Camera&&) = delete;


	public:
		void EnableRender();
		void DisableRender();
		bool Enabled() const;
		void Resize(const size_t& width, const size_t& height);
		void SetPixel(const size_t& x, const size_t& y, const Graphics::Color& color);

		void SetPosition(const Math::vec3<float>& position);
		void SetRotation(const Math::vec3<float>& rotation);
		void SetFov(const Math::angle<Math::rad, float>& fov);

		void SetFocalDistance(float focal_distance);
		void SetAperture(float aperture);


		size_t GetWidth() const;
		size_t GetHeight() const;
		float GetAspectRatio() const;

		const Math::vec3<float>& GetPosition() const;
		const Math::vec3<float>& GetRotation() const;
		const Math::angle<Math::rad, float>& GetFov() const;
		float GetFocalDistance() const;
		float GetAperture() const;
		size_t GetSamplesCount() const;

		const Graphics::Bitmap& GetBitmap() const;

		friend class ObjectCreator;
	};


	template<> struct ConStruct<Camera> : public ConStruct<WorldObject>
	{
		Math::vec3<float> position;
		Math::vec3<float> rotation;
		size_t width, height;
		Math::angle<Math::rad, float> fov;
		float focal_distance;
		float aperture;
		bool enabled;

		ConStruct(ConStruct<WorldObject> con_struct = ConStruct<WorldObject>(),
			Math::vec3<float> position = Math::vec3<float>(0.0f, -10.0f, 0.0f),
			Math::vec3<float> rotation = Math::vec3<float>(0.0f, 0.0f, 0.0f),
			const size_t& width = 800u, const size_t& height = 600u,
			Math::angle<Math::rad, float> fov = Math::constants<float>::pi / 2.0f,
			float focal_distance = 10.0f,
			float aperture = 0.5f,
			bool enabled = true)
			: ConStruct<WorldObject>(con_struct)
			, position(position)
			, rotation(rotation)
			, width(width)
			, height(height)
			, fov(fov)
			, focal_distance(focal_distance)
			, aperture(aperture)
			, enabled(enabled)
		{}
		~ConStruct()
		{}
	};
}

#endif // !CAMERA_H