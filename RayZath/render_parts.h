#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "vec2.h"
#include "angle.h"
#include "bitmap.h"
#include "world_object.h"

namespace RayZath::Engine
{
	struct CoordSystem
	{
	private:
		Math::vec3f x_axis, y_axis, z_axis;


	public:
		CoordSystem();
		CoordSystem(const Math::vec3f& rotation);

	public:
		const Math::vec3f GetXAxis() const;
		const Math::vec3f GetYAxis() const;
		const Math::vec3f GetZAxis() const;

		Math::vec3f TransformForward(const Math::vec3f& v) const;
		Math::vec3f TransformBackward(const Math::vec3f& v) const;
		void ApplyRotation(const Math::vec3f& rotation);
		void LookAt(const Math::vec3f& rotation);
	};
	struct Transformation
	{
	private:
		Math::vec3f m_position, m_rotation, m_center, m_scale;
		CoordSystem m_coord_system;

	public:
		Transformation(
			const Math::vec3f& position,
			const Math::vec3f& rotation,
			const Math::vec3f& center,
			const Math::vec3f& scale);


	public:
		void LookAtPoint(const Math::vec3f& point, const Math::angle_radf& angle = 0.0f);
		void LookInDirection(const Math::vec3f& direction, const Math::angle_radf& angle = 0.0f);

		const Math::vec3f& GetPosition() const;
		const Math::vec3f& GetRotation() const;
		const Math::vec3f& GetCenter() const;
		const Math::vec3f& GetScale() const;
		const CoordSystem& GetCoordSystem() const;

		void SetPosition(const Math::vec3f& position);
		void SetRotation(const Math::vec3f& rotation);
		void SetCenter(const Math::vec3f& center);
		void SetScale(const Math::vec3f& scale);
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


		void Reset(const Math::vec3f& point = Math::vec3f(0.0f, 0.0f, 0.0f));
		void ExtendBy(const Math::vec3f& point);
		void ExtendBy(const BoundingBox& bb);

		Math::vec3f GetCentroid() const noexcept;
	};


	template <typename T>
	struct TextureBuffer
		: public WorldObject
	{
	public:
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
	private:
		Graphics::Buffer2D<T> m_bitmap;
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
			const ConStruct<TextureBuffer<T>>& con_struct)
			: WorldObject(updatable, con_struct)
			, m_bitmap(con_struct.bitmap)
			, m_filter_mode(con_struct.filter_mode)
			, m_address_mode(con_struct.address_mode)
			, m_scale(con_struct.scale)
			, m_rotation(con_struct.rotation)
			, m_translation(con_struct.translation)
		{}


	public:
		TextureBuffer& operator=(const TextureBuffer& bitmap) = delete;
		TextureBuffer& operator=(TextureBuffer&& bitmap) = delete;


	public:
		const Graphics::Buffer2D<T>& GetBitmap() const noexcept
		{
			return m_bitmap;
		}
		FilterMode GetFilterMode() const noexcept
		{
			return m_filter_mode;
		}
		AddressMode GetAddressMode() const noexcept
		{
			return m_address_mode;
		}
		Math::vec2f GetScale() const
		{
			return m_scale;
		}
		Math::angle_radf GetRotation() const
		{
			return m_rotation;
		}
		Math::vec2f GetTranslation() const
		{
			return m_translation;
		}

		void SetBitmap(const Graphics::Buffer2D<T>& bitmap)
		{
			m_bitmap = bitmap;
			GetStateRegister().RequestUpdate();
		}
		void SetFilterMode(const FilterMode filter_mode)
		{
			m_filter_mode = filter_mode;
			GetStateRegister().RequestUpdate();
		}
		void SetAddressMode(const AddressMode address_mode)
		{
			m_address_mode = address_mode;
			GetStateRegister().RequestUpdate();
		}
		void SetScale(const Math::vec2f& scale)
		{
			m_scale = scale;
			GetStateRegister().RequestUpdate();
		}
		void SetRotation(const Math::angle_radf& rotation)
		{
			m_rotation = rotation;
			GetStateRegister().RequestUpdate();
		}
		void SetTranslation(const Math::vec2f& translation)
		{
			m_translation = translation;
			GetStateRegister().RequestUpdate();
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
			const std::string& name = "name",
			const Graphics::Buffer2D<T>& bitmap = Graphics::Buffer2D<T>(16u, 16u),
			const typename TextureBuffer<T>::FilterMode& filter_mode = TextureBuffer<T>::FilterMode::Point,
			const typename TextureBuffer<T>::AddressMode& address_mode = TextureBuffer<T>::AddressMode::Wrap,
			const Math::vec2f& scale = Math::vec2f(1.0f, 1.0f),
			const Math::angle_radf& rotation = Math::angle_radf(0.0f),
			const Math::vec2f& translation = Math::vec2f(0.0f, 0.0f))
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
			, scale(scale)
			, rotation(rotation)
			, translation(translation)
		{}
	};
}

#endif // !RENDER_PARTS_H