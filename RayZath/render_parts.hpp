#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "vec2.h"
#include "angle.h"
#include "bitmap.h"
#include "world_object.hpp"

namespace RayZath::Engine
{
	struct CoordSystem
	{
	private:
		Math::vec3f x_axis, y_axis, z_axis;

	public:
		CoordSystem();
		CoordSystem(const Math::vec3f& rotation);

		CoordSystem& operator*=(const CoordSystem& other);

		const Math::vec3f xAxis() const;
		const Math::vec3f yAxis() const;
		const Math::vec3f zAxis() const;

		Math::vec3f transformForward(const Math::vec3f& v) const;
		Math::vec3f transformBackward(const Math::vec3f& v) const;
		void applyRotation(const Math::vec3f& rotation);
		void lookAt(const Math::vec3f& rotation);
	};
	struct Transformation
	{
	private:
		Math::vec3f m_position, m_rotation, m_scale;
		CoordSystem m_coord_system;

	public:
		Transformation(
			const Math::vec3f& position = Math::vec3f(0.0f),
			const Math::vec3f& rotation = Math::vec3f(0.0f),
			const Math::vec3f& scale = Math::vec3f(1.0f));

		Transformation& operator*=(const Transformation& other);

		void lookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void lookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Math::vec3f& position() const;
		const Math::vec3f& rotation() const;
		const Math::vec3f& scale() const;
		const CoordSystem& coordSystem() const;

		void position(const Math::vec3f& position);
		void rotation(const Math::vec3f& rotation);
		void scale(const Math::vec3f& scale);
	};
	struct BoundingBox
	{
	public:
		Math::vec3f min, max;


	public:
		BoundingBox(
			const Math::vec3f& p1 = Math::vec3f(0.0f, 0.0f, 0.0f),
			const Math::vec3f& p2 = Math::vec3f(0.0f, 0.0f, 0.0f));
		BoundingBox(
			const Math::vec3f& p1,
			const Math::vec3f& p2,
			const Math::vec3f& p3);


		void reset(const Math::vec3f& point = Math::vec3f(0.0f, 0.0f, 0.0f));
		void extendBy(const Math::vec3f& point);
		void extendBy(const BoundingBox& bb);

		Math::vec3f centroid() const noexcept;
	};

	struct TextureBufferBase
	{
		enum class FilterMode
		{
			Point,
			Linear
		};
		enum class AddressMode
		{
			Wrap,
			Clamp,
			Mirror,
			Border
		};
	};

	template <typename T>
	struct TextureBuffer
		: public WorldObject
		, public TextureBufferBase
	{
	public:		
		using buffer_t = Graphics::Buffer2D<T>;
	private:
		buffer_t m_bitmap;
		FilterMode m_filter_mode;
		AddressMode m_address_mode;
		Math::vec2f m_scale;
		Math::angle_radf m_rotation;
		Math::vec2f m_translation;


	public:
		TextureBuffer(const TextureBuffer& bitmap) = delete;
		TextureBuffer(TextureBuffer&& bitmap) = delete;
		TextureBuffer(
			Updatable* updatable,
			ConStruct<TextureBuffer<T>> con_struct)
			: WorldObject(updatable, con_struct)
			, m_bitmap(std::move(con_struct.bitmap))
			, m_filter_mode(std::move(con_struct.filter_mode))
			, m_address_mode(std::move(con_struct.address_mode))
			, m_scale(std::move(con_struct.scale))
			, m_rotation(std::move(con_struct.rotation))
			, m_translation(std::move(con_struct.translation))
		{}


	public:
		TextureBuffer& operator=(const TextureBuffer& bitmap) = delete;
		TextureBuffer& operator=(TextureBuffer&& bitmap) = delete;


	public:
		const auto& bitmap() const noexcept
		{
			return m_bitmap;
		}
		FilterMode filterMode() const noexcept
		{
			return m_filter_mode;
		}
		AddressMode addressMode() const noexcept
		{
			return m_address_mode;
		}
		Math::vec2f scale() const
		{
			return m_scale;
		}
		Math::angle_radf rotation() const
		{
			return m_rotation;
		}
		Math::vec2f translation() const
		{
			return m_translation;
		}

		void bitmap(const buffer_t& bitmap)
		{
			m_bitmap = bitmap;
			stateRegister().RequestUpdate();
		}
		void filterMode(const FilterMode filter_mode)
		{
			m_filter_mode = filter_mode;
			stateRegister().RequestUpdate();
		}
		void addressMode(const AddressMode address_mode)
		{
			m_address_mode = address_mode;
			stateRegister().RequestUpdate();
		}
		void scale(const Math::vec2f& scale)
		{
			m_scale = scale;
			stateRegister().RequestUpdate();
		}
		void rotation(const Math::angle_radf& rotation)
		{
			m_rotation = rotation;
			stateRegister().RequestUpdate();
		}
		void translation(const Math::vec2f& translation)
		{
			m_translation = translation;
			stateRegister().RequestUpdate();
		}

	};
	typedef TextureBuffer<Graphics::Color> Texture;
	typedef TextureBuffer<Graphics::Color> NormalMap;
	typedef TextureBuffer<uint8_t> MetalnessMap;
	typedef TextureBuffer<uint8_t> RoughnessMap;
	typedef TextureBuffer<float> EmissionMap;
	
	template <typename T>
	struct ConStruct<TextureBuffer<T>>
		: public ConStruct<WorldObject>
	{
		Graphics::Buffer2D<T> bitmap;
		typename TextureBuffer<T>::FilterMode filter_mode;
		typename TextureBuffer<T>::AddressMode address_mode;
		Math::vec2f scale;
		Math::angle_radf rotation;
		Math::vec2f translation;

		ConStruct(
			std::string name = "name",
			Graphics::Buffer2D<T> bitmap = Graphics::Buffer2D<T>(16u, 16u),
			typename TextureBuffer<T>::FilterMode filter_mode = TextureBuffer<T>::FilterMode::Point,
			typename TextureBuffer<T>::AddressMode address_mode = TextureBuffer<T>::AddressMode::Wrap,
			Math::vec2f scale = Math::vec2f(1.0f, 1.0f),
			Math::angle_radf rotation = Math::angle_radf(0.0f),
			Math::vec2f translation = Math::vec2f(0.0f, 0.0f))
			: ConStruct<WorldObject>(name)
			, bitmap(std::move(bitmap))
			, filter_mode(std::move(filter_mode))
			, address_mode(std::move(address_mode))
			, scale(std::move(scale))
			, rotation(std::move(rotation))
			, translation(std::move(translation))
		{}
	};

	struct Material;
	struct Ray
	{
	public:
		Math::vec3f32 origin;
		Math::vec3f32 direction;

	public:
		Ray() = default;
		Ray(const Math::vec3f32 origin, const Math::vec3f32 direction)
			: origin(origin)
			, direction(direction)
		{
			this->direction.Normalize();
		}
	};
	struct RangedRay : public Ray
	{
	public:
		Math::vec2f32 near_far;

	public:
		RangedRay()
			: near_far(0.0f, std::numeric_limits<std::float_t>().max())
		{}
		RangedRay(
			const Math::vec3f32& origin,
			const Math::vec3f32& direction,
			const Math::vec2f32 near_far = Math::vec2f32(0.0f, std::numeric_limits<std::float_t>().max()))
			: Ray(origin, direction)
			, near_far(near_far)
		{}

	public:
		void resetRange(const Math::vec2f32 range = Math::vec2f32(0.0f, std::numeric_limits<std::float_t>().max()))
		{
			near_far = range;
		}
	};
	struct SceneRay : public RangedRay
	{
	public:
		const Material* material;
		Graphics::ColorF color;


	public:
		SceneRay()
			: RangedRay()
			, material(nullptr)
			, color(1.0f)
		{}
		SceneRay(
			const Math::vec3f32& origin,
			const Math::vec3f32& direction,
			const Material* material,
			const Graphics::ColorF& color = Graphics::ColorF(1.0f),
			const Math::vec2f32 near_far = Math::vec2f32(0.0f, std::numeric_limits<std::float_t>().max()))
			: RangedRay(origin, direction, near_far)
			, material(material)
			, color(color)
		{}
	};
}

#endif // !RENDER_PARTS_H