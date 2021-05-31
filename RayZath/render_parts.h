#ifndef RENDER_PARTS_H
#define RENDER_PARTS_H

#include "vec3.h"
#include "vec2.h"
#include "angle.h"
#include "bitmap.h"
#include "world_object.h"

namespace RayZath
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

	typedef Math::vec2f Texcrd;

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
		enum class OriginPosition
		{
			TopLeft,
			TopRight,
			BottomLeft,
			BottomRight
		};
	private:
		Graphics::Buffer2D<T> m_bitmap;
		FilterMode m_filter_mode;
		AddressMode m_address_mode;
		OriginPosition m_origin_position;


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
			, m_origin_position(con_struct.origin_position)
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
		OriginPosition GetOriginPosition() const
		{
			return m_origin_position;
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
		void SetOriginPosition(const OriginPosition origin_position)
		{
			m_origin_position = origin_position;
			GetStateRegister().RequestUpdate();
		}
	};
	typedef TextureBuffer<Graphics::Color> Texture;
	typedef TextureBuffer<Graphics::Color> NormalMap;
	typedef TextureBuffer<float> EmittanceMap;
	typedef TextureBuffer<uint8_t> ReflectanceMap;
	
	/*template <typename T>
	struct ConStruct<TextureBuffer<T>>
		: public ConStruct<WorldObject>
	{
		Graphics::Buffer2D<T> bitmap;
		TextureBuffer<T>::FilterMode filter_mode;
		TextureBuffer<T>::AddressMode address_mode;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Buffer2D<T>& bitmap = Graphics::Buffer2D<T>(16u, 16u),
			const TextureBuffer<T>::FilterMode& filter_mode = TextureBuffer<T>::FilterMode::Point,
			const TextureBuffer<T>::AddressMode& address_mode = TextureBuffer<T>::AddressMode::Wrap)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
		{}
	};*/
	template <>
	struct ConStruct<Texture>
		: public ConStruct<WorldObject>
	{
		Graphics::Bitmap bitmap;
		Texture::FilterMode filter_mode;
		Texture::AddressMode address_mode;
		Texture::OriginPosition origin_position;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Bitmap& bitmap = Graphics::Bitmap(64u, 64u),
			const Texture::FilterMode& filter_mode = Texture::FilterMode::Point,
			const Texture::AddressMode& address_mode = Texture::AddressMode::Wrap,
			const Texture::OriginPosition& origin_position = Texture::OriginPosition::BottomLeft)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
			, origin_position(origin_position)
		{}
	};
	template <>
	struct ConStruct<EmittanceMap>
		: public ConStruct<WorldObject>
	{
		Graphics::Buffer2D<float> bitmap;
		EmittanceMap::FilterMode filter_mode;
		EmittanceMap::AddressMode address_mode;
		EmittanceMap::OriginPosition origin_position;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Buffer2D<float>& bitmap = Graphics::Buffer2D<float>(64u, 64u),
			const EmittanceMap::FilterMode& filter_mode = EmittanceMap::FilterMode::Point,
			const EmittanceMap::AddressMode& address_mode = EmittanceMap::AddressMode::Wrap,
			const EmittanceMap::OriginPosition& origin_position = EmittanceMap::OriginPosition::BottomLeft)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
			, origin_position(origin_position)
		{}
	};
	template <>
	struct ConStruct<ReflectanceMap>
		: public ConStruct<WorldObject>
	{
		Graphics::Buffer2D<uint8_t> bitmap;
		ReflectanceMap::FilterMode filter_mode;
		ReflectanceMap::AddressMode address_mode;
		ReflectanceMap::OriginPosition origin_position;

		ConStruct(
			const std::wstring& name = L"name",
			const Graphics::Buffer2D<uint8_t>& bitmap = Graphics::Buffer2D<uint8_t>(64u, 64u),
			const ReflectanceMap::FilterMode& filter_mode = ReflectanceMap::FilterMode::Point,
			const ReflectanceMap::AddressMode& address_mode = ReflectanceMap::AddressMode::Wrap,
			const ReflectanceMap::OriginPosition& origin_position = ReflectanceMap::OriginPosition::BottomLeft)
			: ConStruct<WorldObject>(name)
			, bitmap(bitmap)
			, filter_mode(filter_mode)
			, address_mode(address_mode)
			, origin_position(origin_position)
		{}
	};



	typedef Math::vec3f Vertex;
	typedef Math::vec3f Normal;
	struct Triangle
	{
	public:
		Vertex *v1, *v2, *v3;
		Texcrd *t1, *t2, *t3;
		Normal *n1, *n2, *n3;
		Math::vec3f normal;
		uint32_t material_id;


	public:
		Triangle(
			Vertex* v1, Vertex* v2, Vertex* v3,
			Texcrd* t1, Texcrd* t2, Texcrd* t3,
			Normal* n1, Normal* n2, Normal* n3,
			const uint32_t& mat_id);
		~Triangle();


	public:
		void CalculateNormal();
		BoundingBox GetBoundingBox() const;
	};
}

#endif // !RENDER_PARTS_H